package("libtorch-bin")
    set_kind("library")
    set_homepage("https://pytorch.org/cppdocs/")
    set_description("Prebuilt libtorch binary distribution from PyTorch")
    set_license("BSD-3-Clause")

    local variant = os.getenv("TYPETORCH_LIBTORCH_VARIANT") or "cu128"
    if variant == "cpu" then
        add_urls("https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.8.0%2Bcpu.zip",
                 {version = "2.8.0+cpu"})
    else
        add_urls("https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.8.0%2Bcu128.zip",
                 {version = "2.8.0+cu128"})
    end

    on_load(function (package)
        package:add("includedirs", "include")
        package:add("includedirs", "include/torch/csrc/api/include")
        package:add("linkdirs", "lib")
        package:add("links", "torch", "torch_cpu", "c10")
        package:add("rpathdirs", "lib")
        package:add("syslinks", "pthread", "dl")
    end)

    on_install("linux", function (package)
        os.cp("include", package:installdir())
        os.cp("lib", package:installdir())
        os.cp("share", package:installdir())
        assert(os.isfile(path.join(package:installdir(), "include", "ATen", "ATen.h")))
    end)
