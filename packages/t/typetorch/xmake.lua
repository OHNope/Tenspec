local module_files = {
    "src/fast_io.mpp",
    "src/fast_io.cpp",
    "src/libtorch.mpp",
    "src/typetorch/types.mpp",
    "src/typetorch/torch_mappings.mpp",
    "src/typetorch/shape_meta.mpp",
    "src/typetorch/common.mpp",
    "src/typetorch/runtime_checks.mpp",
    "src/typetorch/tensor/core.mpp",
    "src/typetorch/tensor/factory.mpp",
    "src/typetorch/tensor/arithmetic.mpp",
    "src/typetorch/tensor/view.mpp",
    "src/typetorch/tensor/nn.mpp",
    "src/typetorch/tensor/tensor.mpp",
    "src/typetorch/nnModules/core.mpp",
    "src/typetorch/nnModules/linear.mpp",
    "src/typetorch/nnModules/embedding.mpp",
    "src/typetorch/nnModules/layer_norm.mpp",
    "src/typetorch/nnModules/rms_norm.mpp",
    "src/typetorch/nnModules/nnModules.mpp",
    "src/typetorch/typetorch.mpp",
}

package("typetorch")
    set_kind("library", {headeronly = true})
    set_description("Type-safe LibTorch tensor wrappers with C++26 compile-time contracts")
    set_license("MIT")
    set_sourcedir(path.join(os.scriptdir(), "../../.."))

    local libtorch_variant = os.getenv("TYPETORCH_LIBTORCH_VARIANT") or "cu128"
    local libtorch_version = libtorch_variant == "cpu" and "2.8.0+cpu" or "2.8.0+cu128"
    add_deps("libtorch-bin " .. libtorch_version, {alias = "libtorch"})

    on_install(function (package)
        local moddir = package:installdir("modules")
        local incdir = package:installdir("include")

        os.cp("third_party/fast_io/include", package:installdir())
        os.cp("src/attributes.inc", incdir)
        os.cp("src/type_macros.inc", incdir)
        os.mkdir(path.join(incdir, "typetorch", "nnModules"))
        os.cp("src/typetorch/nnModules/smartptr.inc",
              path.join(incdir, "typetorch", "nnModules", "smartptr.inc"))

        for _, file in ipairs(module_files) do
            local dst = path.join(moddir, file)
            os.mkdir(path.directory(dst))
            os.cp(file, dst)
        end
    end)

    on_load(function (package)
        package:add("includedirs", "include", "modules")
        package:add("cxxflags", "-Wall", "-Wextra", "-fvisibility=hidden")
    end)
package_end()
