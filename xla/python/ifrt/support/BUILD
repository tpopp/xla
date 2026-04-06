load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla/python/ifrt:internal"]),
    licenses = ["notice"],
)

cc_library(
    name = "module_parsing",
    hdrs = ["module_parsing.h"],
    compatible_with = get_compatible_with_portable(),
    deprecation = "Use //third_party/tensorflow/compiler/xla/python/ifrt/ir/support:module_parsing instead.",
    visibility = internal_visibility([
        "//xla/python/ifrt/ir:users",
    ]),
    deps = [
        "//xla/python/ifrt/ir/support:module_parsing",
    ],
)
