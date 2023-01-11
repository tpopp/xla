/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/python/profiler.h"

#include <memory>

#include "third_party/jsoncpp/json.h"
#include "pybind11/pybind11.h"
#include "xla/python/profiler/internal/traceme_wrapper.h"
#include "xla/python/types.h"
#include "xla/status.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/rpc/client/capture_profile.h"
#include "tsl/profiler/rpc/profiler_server.h"
#include "tsl/profiler/utils/tf_xplane_visitor.h"
#include "tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/utils/xplane_visitor.h"

namespace xla {

using tensorflow::profiler::XPlane;
using tensorflow::profiler::XSpace;
using tensorflow::profiler::XStat;
using tsl::profiler::CreateTfXPlaneVisitor;
using tsl::profiler::FindPlanesWithPrefix;
using tsl::profiler::FindPlaneWithName;
using tsl::profiler::HostEventType;
using tsl::profiler::IsInternalEvent;
using tsl::profiler::IsInternalStat;
using tsl::profiler::kCustomPlanePrefix;
using tsl::profiler::kFirstDeviceId;
using tsl::profiler::kGpuPlanePrefix;
using tsl::profiler::kHostThreadsDeviceId;
using tsl::profiler::kHostThreadsPlaneName;
using tsl::profiler::kTpuPlanePrefix;
using tsl::profiler::StatType;
using tsl::profiler::XEventVisitor;
using tsl::profiler::XLineVisitor;
using tsl::profiler::XPlaneVisitor;
using tsl::profiler::XStatVisitor;
namespace {

// Converts the given time from picoseconds to microseconds and then to a string
// using maximum precision.
inline std::string PicosToMicrosString(uint64 ps) {
  char buffer[32];
  int result =
      snprintf(buffer, sizeof(buffer), "%.17g", tsl::profiler::PicoToMicro(ps));
  DCHECK(result > 0 && result < sizeof(buffer));
  return std::string(buffer);
}

// Escapes and quotes the given string.
inline std::string JsonString(const std::string& s) {
  return Json::valueToQuotedString(s.c_str());
}

void BuildDeviceAndResources(uint32 device_id, const XPlaneVisitor& plane,
                             std::string* json) {
  if (!plane.Name().empty()) {
    absl::StrAppend(json, R"({"ph":"M","pid":)", device_id,
                    R"(,"name":"process_name","args":{"name":)",
                    JsonString(std::string(plane.Name())), "}},");
  }
  absl::StrAppend(json, R"({"ph":"M","pid":)", device_id,
                  R"(,"name":"process_sort_index","args":{"sort_index":)",
                  device_id, "}},");

  bool sort_by_ordinal = (device_id == tsl::profiler::kHostThreadsDeviceId);
  int ordinal = 0;
  plane.ForEachLine([&](const XLineVisitor& line) {
    uint32 resource_id = line.DisplayId();
    uint32 sort_index = resource_id;
    if (sort_by_ordinal) {
      // When sort_index is absent (i.e. 0), resource id will be used.
      // Therefore sort_index starts with 1.
      sort_index = ++ordinal;
    }
    if (!line.DisplayName().empty()) {
      absl::StrAppend(json, R"({"ph":"M","pid":)", device_id, R"(,"tid":)",
                      resource_id, R"(,"name":"thread_name","args":{"name":)",
                      JsonString(std::string(line.DisplayName())), "}},");
    }
    absl::StrAppend(json, R"({"ph":"M","pid":)", device_id, R"(,"tid":)",
                    resource_id, R"(,"name":"thread_sort_index")",
                    R"(,"args":{"sort_index":)", sort_index, "}},");
  });
}

void ConvertXPlaneToTraceEvents(uint32 device_id, const XPlaneVisitor& xplane,
                                uint64_t cutoff_timestamp, std::string* json) {
  // Convert devices and resources.
  BuildDeviceAndResources(device_id, xplane, json);

  // Convert events.
  xplane.ForEachLine([device_id, cutoff_timestamp,
                      json](const XLineVisitor& xline) {
    uint32 resource_id = xline.DisplayId();
    if (xline.DisplayName() == tsl::profiler::kXlaAsyncOpLineName) {
      return;
    }
    xline.ForEachEvent([device_id, resource_id, cutoff_timestamp,
                        json](const XEventVisitor& xevent) {
      int64_t event_type =
          xevent.Type().value_or(HostEventType::kUnknownHostEventType);
      if (IsInternalEvent(event_type)) return;
      if (xevent.TimestampPs() > cutoff_timestamp) return;
      std::vector<std::pair<std::string, std::string>> args;
      std::string name;
      if (xevent.HasDisplayName()) {
        name = std::string(xevent.DisplayName());
        args.emplace_back("long_name", std::string(xevent.Name()));
      } else {
        name = std::string(xevent.Name());
      }

      auto for_each_stat = [&](const XStatVisitor& stat) {
        if (stat.ValueCase() == XStat::VALUE_NOT_SET) return;
        if (IsInternalStat(stat.Type())) return;
        if (stat.Type() == StatType::kStepName) {
          name = stat.ToString();
        }
        args.emplace_back(std::string(stat.Name()), stat.ToString());
      };
      // The metadata stats should appear before the per-occurrence stats.
      xevent.Metadata().ForEachStat(for_each_stat);
      xevent.ForEachStat(for_each_stat);
      absl::c_sort(args, [](const std::pair<std::string, std::string>& a,
                            const std::pair<std::string, std::string>& b) {
        return a.first < b.first;
      });

      auto duration_ps = std::max(xevent.DurationPs(), int64_t{1});
      absl::StrAppend(
          json, R"({"ph":"X","pid":)", device_id, R"(,"tid":)", resource_id,
          R"(,"ts":)", PicosToMicrosString(xevent.TimestampPs()), R"(,"dur":)",
          PicosToMicrosString(duration_ps), R"(,"name":)", JsonString(name));
      if (!args.empty()) {
        absl::StrAppend(json, R"(,"args":{)");
        for (const auto& arg : args) {
          absl::StrAppend(json, JsonString(arg.first), ":",
                          JsonString(arg.second), ",");
        }
        // Replace trailing comma with closing brace.
        json->back() = '}';
      }
      absl::StrAppend(json, "},");
    });
  });
}

std::vector<std::pair<uint32_t, const XPlane*>> FindOutputPlanes(
    const XSpace& xspace) {
  std::vector<std::pair<uint32_t, const XPlane*>> output_planes;
  const XPlane* host_plane = FindPlaneWithName(xspace, kHostThreadsPlaneName);
  if (host_plane != nullptr) {
    output_planes.emplace_back(kHostThreadsDeviceId, host_plane);
  }
  std::vector<const XPlane*> device_planes =
      FindPlanesWithPrefix(xspace, kGpuPlanePrefix);
  // We don't expect GPU and TPU planes and custom devices to be present in the
  // same XSpace.
  if (device_planes.empty()) {
    device_planes = FindPlanesWithPrefix(xspace, kTpuPlanePrefix);
  }
  if (device_planes.empty()) {
    device_planes = FindPlanesWithPrefix(xspace, kCustomPlanePrefix);
  }
  for (const XPlane* device_plane : device_planes) {
    XPlaneVisitor xplane = CreateTfXPlaneVisitor(device_plane);
    uint32 device_id = kFirstDeviceId + xplane.Id();
    output_planes.emplace_back(device_id, device_plane);
  }
  return output_planes;
}

uint64_t FindCuttoffTimestamp(
    std::vector<std::pair<uint32_t, const XPlane*>>& output_planes,
    size_t limit) {
  size_t num_events = 0;
  for (auto plane : output_planes) {
    XPlaneVisitor xplane = CreateTfXPlaneVisitor(plane.second);
    xplane.ForEachLine([&num_events](const XLineVisitor& xline) {
      num_events += xline.NumEvents();
    });
  }
  uint64 cutoff_timestamp = std::numeric_limits<uint64_t>::max();
  if (num_events > limit) {
    std::vector<uint64> timestamps;
    timestamps.reserve(num_events);
    for (auto plane : output_planes) {
      XPlaneVisitor xplane = CreateTfXPlaneVisitor(plane.second);
      xplane.ForEachLine([&timestamps](const XLineVisitor& xline) {
        xline.ForEachEvent([&timestamps](const XEventVisitor& xevent) {
          timestamps.push_back(xevent.TimestampPs());
        });
      });
    }
    std::partial_sort(timestamps.begin(), timestamps.begin() + limit,
                      timestamps.end(), std::less<uint64>());
    cutoff_timestamp = timestamps[limit - 1];
  }
  return cutoff_timestamp;
}

std::string MakeXSpaceJson(const XSpace& xspace) {
  std::string json = R"({"displayTimeUnit":"ns","traceEvents":[)";
  auto output_planes = FindOutputPlanes(xspace);
  constexpr size_t kMaxNumEvents = 1000000;
  uint64_t cutoff_timestamp =
      FindCuttoffTimestamp(output_planes, kMaxNumEvents);
  for (auto plane : output_planes) {
    XPlaneVisitor xplane = CreateTfXPlaneVisitor(plane.second);
    ConvertXPlaneToTraceEvents(plane.first, xplane, cutoff_timestamp, &json);
  }
  // Add one fake event to avoid dealing with no-trailing-comma rule.
  absl::StrAppend(&json, "{}]}");
  return json;
}

}  // namespace

namespace py = pybind11;

namespace {
// Adds a trivial forwarding class so these Python bindings and TensorFlow's
// bindings of the same thing don't register the same class with pybind11.
class TraceMeWrapper : public xla::profiler::TraceMeWrapper {
 public:
  using xla::profiler::TraceMeWrapper::TraceMeWrapper;
};

tensorflow::ProfileOptions DefaultPythonProfileOptions() {
  tensorflow::ProfileOptions options = tsl::ProfilerSession::DefaultOptions();
  options.set_python_tracer_level(1);
  options.set_enable_hlo_proto(true);
  return options;
}
}  // namespace

void BuildProfilerSubmodule(py::module* m) {
  py::module profiler =
      m->def_submodule("profiler", "TensorFlow profiler integration");
  py::class_<tsl::profiler::ProfilerServer,
             std::unique_ptr<tsl::profiler::ProfilerServer>>
      profiler_server_class(profiler, "ProfilerServer");
  profiler.def(
      "start_server",
      [](int port) -> std::unique_ptr<tsl::profiler::ProfilerServer> {
        auto server = std::make_unique<tsl::profiler::ProfilerServer>();
        server->StartProfilerServer(port);
        return server;
      },
      py::arg("port"));

  py::class_<tsl::ProfilerSession> profiler_session_class(profiler,
                                                          "ProfilerSession");
  profiler_session_class
      .def(py::init([]() {
        return tsl::ProfilerSession::Create(DefaultPythonProfileOptions());
      }))
      .def(py::init([](const tensorflow::ProfileOptions& options) {
        return tsl::ProfilerSession::Create(options);
      }))
      .def("stop_and_export",
           [](tsl::ProfilerSession* sess, const std::string& tensorboard_dir,
              bool export_json) -> xla::StatusOr<std::optional<py::bytes>> {
             tensorflow::profiler::XSpace xspace;
             // Disables the ProfilerSession
             TF_RETURN_IF_ERROR(sess->CollectData(&xspace));
             TF_RETURN_IF_ERROR(
                 tsl::profiler::ExportToTensorBoard(xspace, tensorboard_dir));
             if (export_json) {
               return std::optional<py::bytes>(
                   py::bytes(MakeXSpaceJson(xspace)));
             }
             return std::optional<py::bytes>(std::nullopt);
           });

  py::class_<tensorflow::ProfileOptions> profile_options_class(
      profiler, "ProfileOptions");
  profile_options_class.def(py::init(&DefaultPythonProfileOptions))
      .def_property("include_dataset_ops",
                    &tensorflow::ProfileOptions::include_dataset_ops,
                    &tensorflow::ProfileOptions::set_include_dataset_ops)
      .def_property("host_tracer_level",
                    &tensorflow::ProfileOptions::host_tracer_level,
                    &tensorflow::ProfileOptions::set_host_tracer_level)
      .def_property("python_tracer_level",
                    &tensorflow::ProfileOptions::python_tracer_level,
                    &tensorflow::ProfileOptions::set_python_tracer_level)
      .def_property("enable_hlo_proto",
                    &tensorflow::ProfileOptions::enable_hlo_proto,
                    &tensorflow::ProfileOptions::set_enable_hlo_proto)
      .def_property("start_timestamp_ns",
                    &tensorflow::ProfileOptions::start_timestamp_ns,
                    &tensorflow::ProfileOptions::set_start_timestamp_ns)
      .def_property("duration_ms", &tensorflow::ProfileOptions::duration_ms,
                    &tensorflow::ProfileOptions::set_duration_ms)
      .def_property(
          "repository_path", &tensorflow::ProfileOptions::repository_path,
          [](tensorflow::ProfileOptions* options, const std::string& path) {
            options->set_repository_path(path);
          });

  py::class_<TraceMeWrapper> traceme_class(profiler, "TraceMe",
                                           py::module_local());
  traceme_class.def(py::init<py::str, py::kwargs>())
      .def("__enter__", [](py::object self) -> py::object { return self; })
      .def("__exit__",
           [](py::object self, const py::object& ex_type,
              const py::object& ex_value,
              const py::object& traceback) -> py::object {
             py::cast<TraceMeWrapper*>(self)->Stop();
             return py::none();
           })
      .def("set_metadata", &TraceMeWrapper::SetMetadata)
      .def_static("is_enabled", &TraceMeWrapper::IsEnabled);
}

}  // namespace xla
