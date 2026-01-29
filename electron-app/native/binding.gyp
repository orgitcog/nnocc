{
  "targets": [
    {
      "target_name": "cognitive-addon",
      "sources": [
        "cognitive-addon.cc",
        "atomspace-bridge.cc",
        "attention-bridge.cc",
        "inference-bridge.cc"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "../../../cogutil",
        "../../../atomspace",
        "../../../attention",
        "../../../pln",
        "../../../ure"
      ],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc!": [ "-fno-exceptions" ],
      "defines": [ "NAPI_DISABLE_CPP_EXCEPTIONS" ],
      "conditions": [
        ["OS=='win'", {
          "defines": [
            "_WIN32",
            "_USE_MATH_DEFINES",
            "NOMINMAX"
          ],
          "msvs_settings": {
            "VCCLCompilerTool": {
              "ExceptionHandling": 1,
              "AdditionalOptions": [ "/std:c++17" ]
            }
          },
          "libraries": [
            "-l../../build/Release/cogutil.lib",
            "-l../../build/Release/atomspace.lib",
            "-l../../build/Release/attention.lib"
          ]
        }],
        ["OS=='linux'", {
          "cflags_cc": [ "-std=c++17", "-fexceptions" ],
          "libraries": [
            "-L../../build",
            "-lcogutil",
            "-latomspace",
            "-lattention"
          ]
        }],
        ["OS=='mac'", {
          "xcode_settings": {
            "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
            "CLANG_CXX_LIBRARY": "libc++",
            "MACOSX_DEPLOYMENT_TARGET": "10.15",
            "OTHER_CFLAGS": [ "-std=c++17" ]
          },
          "libraries": [
            "-L../../build",
            "-lcogutil",
            "-latomspace",
            "-lattention"
          ]
        }]
      ]
    }
  ]
}
