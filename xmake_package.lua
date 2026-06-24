-- xmake_package.lua — Include this from consumer projects to use typetorch
-- via add_requires() + add_rules("typetorch.modules").
--
-- Usage:
--   add_repositories("typetorch-repo path/to/typetorch")
--   includes("path/to/typetorch/xmake_package.lua")
--   add_requires("typetorch")
--
--   target("my_app")
--       add_packages("typetorch")
--       add_rules("typetorch.modules")
--       add_files("src/main.cpp")

local typetorch_module_files = {
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
    "src/typetorch/nnModules/conv2d.mpp",
    "src/typetorch/nnModules/embedding.mpp",
    "src/typetorch/nnModules/layer_norm.mpp",
    "src/typetorch/nnModules/rms_norm.mpp",
    "src/typetorch/nnModules/nnModules.mpp",
    "src/typetorch/typetorch.mpp",
}

rule("typetorch.modules")
    after_load(function (target)
        local pkg = target:pkg("typetorch")
        if not pkg then
            return
        end
        target:add("cxxflags", "-fmodules", {force = true})
        target:add("cxxflags", "-freflection", {force = true})
        local relocate_ldflags = os.getenv("TYPETORCH_GCC_RELOCATE_LDFLAGS")
        if relocate_ldflags and relocate_ldflags ~= "" then
            for flag in relocate_ldflags:gmatch("%S+") do
                target:add("ldflags", flag, {force = true})
                target:add("shflags", flag, {force = true})
            end
        end
        for _, linkdir in ipairs(pkg:get("linkdirs") or {})
        do
            if os.isfile(path.join(linkdir, "libtorch.so")) then
                target:add("rpathdirs", linkdir)
            end
        end
        local moddir = path.join(pkg:installdir(), "modules")
        if not os.isdir(moddir) then
            return
        end
        for _, file in ipairs(typetorch_module_files) do
            target:add("files", path.join(moddir, file))
        end
    end)
rule_end()
