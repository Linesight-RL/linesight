.. wall_text:

:orphan:

.. code-block:: bash

    ██╗░░░░░██╗███╗░░██╗███████╗░██████╗██╗░██████╗░██╗░░██╗████████╗
    ██║░░░░░██║████╗░██║██╔════╝██╔════╝██║██╔════╝░██║░░██║╚══██╔══╝
    ██║░░░░░██║██╔██╗██║█████╗░░╚█████╗░██║██║░░██╗░███████║░░░██║░░░
    ██║░░░░░██║██║╚████║██╔══╝░░░╚═══██╗██║██║░░╚██╗██╔══██║░░░██║░░░
    ███████╗██║██║░╚███║███████╗██████╔╝██║╚██████╔╝██║░░██║░░░██║░░░
    ╚══════╝╚═╝╚═╝░░╚══╝╚══════╝╚═════╝░╚═╝░╚═════╝░╚═╝░░╚═╝░░░╚═╝░░░


    Training is starting!
    OptimizedModule(
      (_orig_mod): IQN_Network(
        (img_head): Sequential(
          (0): Conv2d(1, 16, kernel_size=(4, 4), stride=(2, 2))
          (1): LeakyReLU(negative_slope=0.01, inplace=True)
          (2): Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2))
          (3): LeakyReLU(negative_slope=0.01, inplace=True)
          (4): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2))
          (5): LeakyReLU(negative_slope=0.01, inplace=True)
          (6): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
          (7): LeakyReLU(negative_slope=0.01, inplace=True)
          (8): Flatten(start_dim=1, end_dim=-1)
        )
        (float_feature_extractor): Sequential(
          (0): Linear(in_features=184, out_features=256, bias=True)
          (1): LeakyReLU(negative_slope=0.01, inplace=True)
          (2): Linear(in_features=256, out_features=256, bias=True)
          (3): LeakyReLU(negative_slope=0.01, inplace=True)
        )
        (A_head): Sequential(
          (0): Linear(in_features=5888, out_features=512, bias=True)
          (1): LeakyReLU(negative_slope=0.01, inplace=True)
          (2): Linear(in_features=512, out_features=12, bias=True)
        )
        (V_head): Sequential(
          (0): Linear(in_features=5888, out_features=512, bias=True)
          (1): LeakyReLU(negative_slope=0.01, inplace=True)
          (2): Linear(in_features=512, out_features=1, bias=True)
        )
        (iqn_fc): Sequential(
          (0): Linear(in_features=64, out_features=5888, bias=True)
          (1): LeakyReLU(negative_slope=0.01, inplace=True)
        )
      )
    )
    +--------------------------------------------+------------+
    |                  Modules                   | Parameters |
    +--------------------------------------------+------------+
    |        _orig_mod.img_head.0.weight         |    256     |
    |         _orig_mod.img_head.0.bias          |     16     |
    |        _orig_mod.img_head.2.weight         |    8192    |
    |         _orig_mod.img_head.2.bias          |     32     |
    |        _orig_mod.img_head.4.weight         |   18432    |
    |         _orig_mod.img_head.4.bias          |     64     |
    |        _orig_mod.img_head.6.weight         |   18432    |
    |         _orig_mod.img_head.6.bias          |     32     |
    | _orig_mod.float_feature_extractor.0.weight |   47104    |
    |  _orig_mod.float_feature_extractor.0.bias  |    256     |
    | _orig_mod.float_feature_extractor.2.weight |   65536    |
    |  _orig_mod.float_feature_extractor.2.bias  |    256     |
    |         _orig_mod.A_head.0.weight          |  3014656   |
    |          _orig_mod.A_head.0.bias           |    512     |
    |         _orig_mod.A_head.2.weight          |    6144    |
    |          _orig_mod.A_head.2.bias           |     12     |
    |         _orig_mod.V_head.0.weight          |  3014656   |
    |          _orig_mod.V_head.0.bias           |    512     |
    |         _orig_mod.V_head.2.weight          |    512     |
    |          _orig_mod.V_head.2.bias           |     1      |
    |         _orig_mod.iqn_fc.0.weight          |   376832   |
    |          _orig_mod.iqn_fc.0.bias           |    5888    |
    +--------------------------------------------+------------+
    Total Trainable Params: 6578333
     Learner could not load weights
     Learner could not load stats
     Could not load optimizer
    All rollout queues were empty. Learner sleeps 1 second.
    All rollout queues were empty. Learner sleeps 1 second.
    Worker could not load weights, exception: [Errno 2] No such file or directory: '/mnt/ext4_data/projects/trackmania_rl/save/run_name/weights1.torch'
    Worker could not load weights, exception: [Errno 2] No such file or directory: '/mnt/ext4_data/projects/trackmania_rl/save/run_name/weights1.torch'
    Worker could not load weights, exception: [Errno 2] No such file or directory: '/mnt/ext4_data/projects/trackmania_rl/save/run_name/weights1.torch'
    Worker could not load weights, exception: [Errno 2] No such file or directory: '/mnt/ext4_data/projects/trackmania_rl/save/run_name/weights1.torch'
    All rollout queues were empty. Learner sleeps 1 second.
    All rollout queues were empty. Learner sleeps 1 second.
    All rollout queues were empty. Learner sleeps 1 second.
    All rollout queues were empty. Learner sleeps 1 second.
    All rollout queues were empty. Learner sleeps 1 second.
    Game not found. Restarting TMInterface.
    Game not found. Restarting TMInterface.
    Game not found. Restarting TMInterface.
    Game not found. Restarting TMInterface.
    ERROR: ld.so: object 'libgamemodeauto.so.0' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.
    ERROR: ld.so: object 'libgamemodeauto.so.0' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.
    002c:fixme:winediag:loader_init wine-staging 8.20 is a testing version containing experimental patches.
    002c:fixme:winediag:loader_init Please mention your exact version when filing bug reports on winehq.org.
    0088:fixme:hid:handle_IRP_MN_QUERY_ID Unhandled type 00000005
    0088:fixme:hid:handle_IRP_MN_QUERY_ID Unhandled type 00000005
    0088:fixme:hid:handle_IRP_MN_QUERY_ID Unhandled type 00000005
    0088:fixme:hid:handle_IRP_MN_QUERY_ID Unhandled type 00000005
    All rollout queues were empty. Learner sleeps 1 second.
    ERROR: ld.so: object 'libgamemodeauto.so.0' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.
    ERROR: ld.so: object 'libgamemodeauto.so.0' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.
    0118:fixme:ntdll:NtQuerySystemInformation info_class SYSTEM_PERFORMANCE_INFORMATION
    info:  Game: TmForever.exe
    info:  DXVK: v2.3
    info:  Vulkan: Found vkGetInstanceProcAddr in winevulkan.dll @ 0x75d53760
    info:  Built-in extension providers:
    info:    Win32 WSI
    info:    OpenVR
    info:    OpenXR
    info:  OpenVR: could not open registry key, status 2
    info:  OpenVR: Failed to locate module
    info:  Enabled instance extensions:
    info:    VK_KHR_get_surface_capabilities2
    info:    VK_KHR_surface
    info:    VK_KHR_win32_surface
    warn:  Skipping CPU adapter: llvmpipe (LLVM 15.0.7, 256 bits)
    info:  D3D9: VK_FORMAT_D16_UNORM_S8_UINT -> VK_FORMAT_D24_UNORM_S8_UINT
    info:  NVIDIA GeForce RTX 4070 Ti:
    info:    Driver : NVIDIA 535.129.3
    info:    Memory Heap[0]:
    info:      Size: 12282 MiB
    info:      Flags: 0x1
    info:      Memory Type[1]: Property Flags = 0x1
    info:    Memory Heap[1]:
    info:      Size: 48130 MiB
    info:      Flags: 0x0
    info:      Memory Type[0]: Property Flags = 0x0
    info:      Memory Type[2]: Property Flags = 0x6
    info:      Memory Type[3]: Property Flags = 0xe
    info:    Memory Heap[2]:
    info:      Size: 246 MiB
    info:      Flags: 0x1
    info:      Memory Type[4]: Property Flags = 0x7
    info:  Process set as DPI aware
    info:  Game: TmForever.exe
    info:  DXVK: v2.3
    info:  Vulkan: Found vkGetInstanceProcAddr in winevulkan.dll @ 0x75b23760
    info:  Built-in extension providers:
    info:    Win32 WSI
    info:    OpenVR
    info:    OpenXR
    info:  OpenVR: could not open registry key, status 2
    info:  OpenVR: Failed to locate module
    info:  Enabled instance extensions:
    info:    VK_KHR_get_surface_capabilities2
    info:    VK_KHR_surface
    info:    VK_KHR_win32_surface
    warn:  Skipping CPU adapter: llvmpipe (LLVM 15.0.7, 256 bits)
    info:  D3D9: VK_FORMAT_D16_UNORM_S8_UINT -> VK_FORMAT_D24_UNORM_S8_UINT
    info:  NVIDIA GeForce RTX 4070 Ti:
    info:    Driver : NVIDIA 535.129.3
    info:    Memory Heap[0]:
    info:      Size: 12282 MiB
    info:      Flags: 0x1
    info:      Memory Type[1]: Property Flags = 0x1
    info:    Memory Heap[1]:
    info:      Size: 48130 MiB
    info:      Flags: 0x0
    info:      Memory Type[0]: Property Flags = 0x0
    info:      Memory Type[2]: Property Flags = 0x6
    info:      Memory Type[3]: Property Flags = 0xe
    info:    Memory Heap[2]:
    info:      Size: 246 MiB
    info:      Flags: 0x1
    info:      Memory Type[4]: Property Flags = 0x7
    info:  Process set as DPI aware
    info:  Device properties:
    info:    Device : NVIDIA GeForce RTX 4070 Ti
    info:    Driver : NVIDIA 535.129.3
    info:  Enabled device extensions:
    info:    VK_EXT_attachment_feedback_loop_layout
    info:    VK_EXT_conservative_rasterization
    info:    VK_EXT_custom_border_color
    info:    VK_EXT_depth_clip_enable
    info:    VK_EXT_extended_dynamic_state3
    info:    VK_EXT_fragment_shader_interlock
    info:    VK_EXT_graphics_pipeline_library
    info:    VK_EXT_memory_priority
    info:    VK_EXT_non_seamless_cube_map
    info:    VK_EXT_robustness2
    info:    VK_EXT_shader_module_identifier
    info:    VK_EXT_transform_feedback
    info:    VK_EXT_vertex_attribute_divisor
    info:    VK_KHR_pipeline_library
    info:    VK_KHR_present_id
    info:    VK_KHR_present_wait
    info:    VK_KHR_swapchain
    info:  Device features:
    info:    robustBufferAccess                     : 1
    info:    fullDrawIndexUint32                    : 1
    info:    imageCubeArray                         : 1
    info:    independentBlend                       : 1
    info:    geometryShader                         : 1
    info:    tessellationShader                     : 0
    info:    sampleRateShading                      : 1
    info:    dualSrcBlend                           : 0
    info:    logicOp                                : 0
    info:    multiDrawIndirect                      : 0
    info:    drawIndirectFirstInstance              : 0
    info:    depthClamp                             : 1
    info:    depthBiasClamp                         : 1
    info:    fillModeNonSolid                       : 1
    info:    depthBounds                            : 1
    info:    wideLines                              : 1
    info:    multiViewport                          : 1
    info:    samplerAnisotropy                      : 1
    info:    textureCompressionBC                   : 1
    info:    occlusionQueryPrecise                  : 1
    info:    pipelineStatisticsQuery                : 1
    info:    vertexPipelineStoresAndAtomics         : 1
    info:    fragmentStoresAndAtomics               : 0
    info:    shaderImageGatherExtended              : 0
    info:    shaderClipDistance                     : 1
    info:    shaderCullDistance                     : 1
    info:    shaderFloat64                          : 0
    info:    shaderInt64                            : 0
    info:    variableMultisampleRate                : 1
    info:    shaderResourceResidency                : 0
    info:    shaderResourceMinLod                   : 0
    info:    sparseBinding                          : 0
    info:    sparseResidencyBuffer                  : 0
    info:    sparseResidencyImage2D                 : 0
    info:    sparseResidencyImage3D                 : 0
    info:    sparseResidency2Samples                : 0
    info:    sparseResidency4Samples                : 0
    info:    sparseResidency8Samples                : 0
    info:    sparseResidency16Samples               : 0
    info:    sparseResidencyAliased                 : 0
    info:  Vulkan 1.1
    info:    shaderDrawParameters                   : 0
    info:  Vulkan 1.2
    info:    samplerMirrorClampToEdge               : 1
    info:    drawIndirectCount                      : 1
    info:    samplerFilterMinmax                    : 0
    info:    hostQueryReset                         : 1
    info:    timelineSemaphore                      : 1
    info:    bufferDeviceAddress                    : 0
    info:    shaderOutputViewportIndex              : 1
    info:    shaderOutputLayer                      : 1
    info:    vulkanMemoryModel                      : 1
    info:  Vulkan 1.3
    info:    robustImageAccess                      : 0
    info:    pipelineCreationCacheControl           : 1
    info:    shaderDemoteToHelperInvocation         : 1
    info:    shaderZeroInitializeWorkgroupMemory    : 0
    info:    synchronization2                       : 1
    info:    dynamicRendering                       : 1
    info:  VK_AMD_shader_fragment_mask
    info:    extension supported                    : 0
    info:  VK_EXT_attachment_feedback_loop_layout
    info:    attachmentFeedbackLoopLayout           : 1
    info:  VK_EXT_conservative_rasterization
    info:    extension supported                    : 1
    info:  VK_EXT_custom_border_color
    info:    customBorderColors                     : 1
    info:    customBorderColorWithoutFormat         : 1
    info:  VK_EXT_depth_clip_enable
    info:    depthClipEnable                        : 1
    info:  VK_EXT_depth_bias_control
    info:    depthBiasControl                       : 0
    info:    leastRepresentableValueForceUnormRepresentation : 0
    info:    floatRepresentation                    : 0
    info:    depthBiasExact                         : 0
    info:  VK_EXT_extended_dynamic_state3
    info:    extDynamicState3AlphaToCoverageEnable  : 1
    info:    extDynamicState3DepthClipEnable        : 1
    info:    extDynamicState3RasterizationSamples   : 1
    info:    extDynamicState3SampleMask             : 1
    info:    extDynamicState3LineRasterizationMode  : 1
    info:  VK_EXT_fragment_shader_interlock
    info:    fragmentShaderSampleInterlock          : 0
    info:    fragmentShaderPixelInterlock           : 0
    info:  VK_EXT_full_screen_exclusive
    info:    extension supported                    : 0
    info:  VK_EXT_graphics_pipeline_library
    info:    graphicsPipelineLibrary                : 1
    info:  VK_EXT_line_rasterization
    info:    rectangularLines                       : 1
    info:    smoothLines                            : 1
    info:  VK_EXT_memory_budget
    info:    extension supported                    : 1
    info:  VK_EXT_memory_priority
    info:    memoryPriority                         : 1
    info:  VK_EXT_non_seamless_cube_map
    info:    nonSeamlessCubeMap                     : 1
    info:  VK_EXT_robustness2
    info:    robustBufferAccess2                    : 1
    info:    robustImageAccess2                     : 1
    info:    nullDescriptor                         : 1
    info:  VK_EXT_shader_module_identifier
    info:    shaderModuleIdentifier                 : 1
    info:  VK_EXT_shader_stencil_export
    info:    extension supported                    : 0
    info:  VK_EXT_swapchain_colorspace
    info:    extension supported                    : 0
    info:  VK_EXT_swapchain_maintenance1
    info:    swapchainMaintenance1                  : 0
    info:  VK_EXT_hdr_metadata
    info:    extension supported                    : 0
    info:  VK_EXT_transform_feedback
    info:    transformFeedback                      : 0
    info:    geometryStreams                        : 0
    info:  VK_EXT_vertex_attribute_divisor
    info:    vertexAttributeInstanceRateDivisor     : 1
    info:    vertexAttributeInstanceRateZeroDivisor : 1
    info:  VK_KHR_external_memory_win32
    info:    extension supported                    : 0
    info:  VK_KHR_external_semaphore_win32
    info:    extension supported                    : 0
    info:  VK_KHR_maintenance5
    info:    maintenance5                           : 0
    info:  VK_KHR_present_id
    info:    presentId                              : 1
    info:  VK_KHR_present_wait
    info:    presentWait                            : 1
    info:  VK_NVX_binary_import
    info:    extension supported                    : 0
    info:  VK_NVX_image_view_handle
    info:    extension supported                    : 0
    info:  VK_KHR_win32_keyed_mutex
    info:    extension supported                    : 0
    info:  Queue families:
    info:    Graphics : 0
    info:    Transfer : 1
    info:    Sparse   : 0
    info:  DXVK: Read 149 valid state cache entries
    info:  DXVK: Graphics pipeline libraries supported
    info:  D3D9DeviceEx: Using extended constant set for software vertex processing.
    info:  D3D9DeviceEx::ResetSwapChain:
    info:    Requested Presentation Parameters
    info:      - Width:              0
    info:      - Height:             0
    info:      - Format:             D3D9Format::A8R8G8B8
    info:      - Auto Depth Stencil: true
    info:                  ^ Format: D3D9Format::D24S8
    info:      - Windowed:           true
    info:      - Swap effect:        1
    info:  Presenter: Actual swap chain properties:
    info:    Format:       VK_FORMAT_B8G8R8A8_UNORM
    info:    Color space:  VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
    info:    Present mode: VK_PRESENT_MODE_IMMEDIATE_KHR (dynamic: no)
    info:    Buffer size:  640x480
    info:    Image count:  3
    info:    Exclusive FS: 0
    warn:  ConvertFormat: Unknown format encountered: 1397249614
    warn:  ConvertFormat: Unknown format encountered: 1397249614
    warn:  ConvertFormat: Unknown format encountered: 1397249614
    warn:  ConvertFormat: Unknown format encountered: 1397249614
    warn:  ConvertFormat: Unknown format encountered: 1397249614
    All rollout queues were empty. Learner sleeps 1 second.
    info:  DXVK: Using 16 compiler threads
    info:  Presenter: Actual swap chain properties:
    info:    Format:       VK_FORMAT_B8G8R8A8_UNORM
    info:    Color space:  VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
    info:    Present mode: VK_PRESENT_MODE_IMMEDIATE_KHR (dynamic: no)
    info:    Buffer size:  640x480
    info:    Image count:  3
    info:    Exclusive FS: 0
    0110:fixme:kernelbase:AppPolicyGetProcessTerminationMethod FFFFFFFA, 0051FEA8
    01ac:fixme:advapi:GetCurrentHwProfileA (0BDCF9B8) semi-stub
    Found Trackmania process id: self.tm_process_id=134656
    Initialize connection to TMInterface
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    ERROR: ld.so: object 'libgamemodeauto.so.0' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    ERROR: ld.so: object 'libgamemodeauto.so.0' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    Connected
    ERROR: ld.so: object 'libgamemodeauto.so.0' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.
    ERROR: ld.so: object 'libgamemodeauto.so.0' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.
    01dc:fixme:ntdll:NtQuerySystemInformation info_class SYSTEM_PERFORMANCE_INFORMATION
    Requested map load
    01e8:fixme:kernelbase:AppPolicyGetThreadInitializationType FFFFFFFA, 1591FF08
    info:  Game: TmForever.exe
    info:  DXVK: v2.3
    info:  Vulkan: Found vkGetInstanceProcAddr in winevulkan.dll @ 0x75b23760
    info:  Built-in extension providers:
    info:    Win32 WSI
    info:    OpenVR
    info:    OpenXR
    info:  OpenVR: could not open registry key, status 2
    info:  OpenVR: Failed to locate module
    info:  Enabled instance extensions:
    info:    VK_KHR_get_surface_capabilities2
    info:    VK_KHR_surface
    info:    VK_KHR_win32_surface
    All rollout queues were empty. Learner sleeps 1 second.
    warn:  Skipping CPU adapter: llvmpipe (LLVM 15.0.7, 256 bits)
    info:  D3D9: VK_FORMAT_D16_UNORM_S8_UINT -> VK_FORMAT_D24_UNORM_S8_UINT
    info:  NVIDIA GeForce RTX 4070 Ti:
    info:    Driver : NVIDIA 535.129.3
    info:    Memory Heap[0]:
    info:      Size: 12282 MiB
    info:      Flags: 0x1
    info:      Memory Type[1]: Property Flags = 0x1
    info:    Memory Heap[1]:
    info:      Size: 48130 MiB
    info:      Flags: 0x0
    info:      Memory Type[0]: Property Flags = 0x0
    info:      Memory Type[2]: Property Flags = 0x6
    info:      Memory Type[3]: Property Flags = 0xe
    info:    Memory Heap[2]:
    info:      Size: 246 MiB
    info:      Flags: 0x1
    info:      Memory Type[4]: Property Flags = 0x7
    info:  Process set as DPI aware
    info:  Game: TmForever.exe
    info:  DXVK: v2.3
    info:  Vulkan: Found vkGetInstanceProcAddr in winevulkan.dll @ 0x75b23760
    info:  Built-in extension providers:
    info:    Win32 WSI
    info:    OpenVR
    info:    OpenXR
    info:  OpenVR: could not open registry key, status 2
    info:  OpenVR: Failed to locate module
    info:  Enabled instance extensions:
    info:    VK_KHR_get_surface_capabilities2
    info:    VK_KHR_surface
    info:    VK_KHR_win32_surface
    warn:  Skipping CPU adapter: llvmpipe (LLVM 15.0.7, 256 bits)
    info:  D3D9: VK_FORMAT_D16_UNORM_S8_UINT -> VK_FORMAT_D24_UNORM_S8_UINT
    info:  NVIDIA GeForce RTX 4070 Ti:
    info:    Driver : NVIDIA 535.129.3
    info:    Memory Heap[0]:
    info:      Size: 12282 MiB
    info:      Flags: 0x1
    info:      Memory Type[1]: Property Flags = 0x1
    info:    Memory Heap[1]:
    info:      Size: 48130 MiB
    info:      Flags: 0x0
    info:      Memory Type[0]: Property Flags = 0x0
    info:      Memory Type[2]: Property Flags = 0x6
    info:      Memory Type[3]: Property Flags = 0xe
    info:    Memory Heap[2]:
    info:      Size: 246 MiB
    info:      Flags: 0x1
    info:      Memory Type[4]: Property Flags = 0x7
    info:  Process set as DPI aware
    info:  Device properties:
    info:    Device : NVIDIA GeForce RTX 4070 Ti
    info:    Driver : NVIDIA 535.129.3
    info:  Enabled device extensions:
    info:    VK_EXT_attachment_feedback_loop_layout
    info:    VK_EXT_conservative_rasterization
    info:    VK_EXT_custom_border_color
    info:    VK_EXT_depth_clip_enable
    info:    VK_EXT_extended_dynamic_state3
    info:    VK_EXT_fragment_shader_interlock
    info:    VK_EXT_graphics_pipeline_library
    info:    VK_EXT_memory_priority
    info:    VK_EXT_non_seamless_cube_map
    info:    VK_EXT_robustness2
    info:    VK_EXT_shader_module_identifier
    info:    VK_EXT_transform_feedback
    info:    VK_EXT_vertex_attribute_divisor
    info:    VK_KHR_pipeline_library
    info:    VK_KHR_present_id
    info:    VK_KHR_present_wait
    info:    VK_KHR_swapchain
    info:  Device features:
    info:    robustBufferAccess                     : 1
    info:    fullDrawIndexUint32                    : 1
    info:    imageCubeArray                         : 1
    info:    independentBlend                       : 1
    info:    geometryShader                         : 1
    info:    tessellationShader                     : 0
    info:    sampleRateShading                      : 1
    info:    dualSrcBlend                           : 0
    info:    logicOp                                : 0
    info:    multiDrawIndirect                      : 0
    info:    drawIndirectFirstInstance              : 0
    info:    depthClamp                             : 1
    info:    depthBiasClamp                         : 1
    info:    fillModeNonSolid                       : 1
    info:    depthBounds                            : 1
    info:    wideLines                              : 1
    info:    multiViewport                          : 1
    info:    samplerAnisotropy                      : 1
    info:    textureCompressionBC                   : 1
    info:    occlusionQueryPrecise                  : 1
    info:    pipelineStatisticsQuery                : 1
    info:    vertexPipelineStoresAndAtomics         : 1
    info:    fragmentStoresAndAtomics               : 0
    info:    shaderImageGatherExtended              : 0
    info:    shaderClipDistance                     : 1
    info:    shaderCullDistance                     : 1
    info:    shaderFloat64                          : 0
    info:    shaderInt64                            : 0
    info:    variableMultisampleRate                : 1
    info:    shaderResourceResidency                : 0
    info:    shaderResourceMinLod                   : 0
    info:    sparseBinding                          : 0
    info:    sparseResidencyBuffer                  : 0
    info:    sparseResidencyImage2D                 : 0
    info:    sparseResidencyImage3D                 : 0
    info:    sparseResidency2Samples                : 0
    info:    sparseResidency4Samples                : 0
    info:    sparseResidency8Samples                : 0
    info:    sparseResidency16Samples               : 0
    info:    sparseResidencyAliased                 : 0
    info:  Vulkan 1.1
    info:    shaderDrawParameters                   : 0
    info:  Vulkan 1.2
    info:    samplerMirrorClampToEdge               : 1
    info:    drawIndirectCount                      : 1
    info:    samplerFilterMinmax                    : 0
    info:    hostQueryReset                         : 1
    info:    timelineSemaphore                      : 1
    info:    bufferDeviceAddress                    : 0
    info:    shaderOutputViewportIndex              : 1
    info:    shaderOutputLayer                      : 1
    info:    vulkanMemoryModel                      : 1
    info:  Vulkan 1.3
    info:    robustImageAccess                      : 0
    info:    pipelineCreationCacheControl           : 1
    info:    shaderDemoteToHelperInvocation         : 1
    info:    shaderZeroInitializeWorkgroupMemory    : 0
    info:    synchronization2                       : 1
    info:    dynamicRendering                       : 1
    info:  VK_AMD_shader_fragment_mask
    info:    extension supported                    : 0
    info:  VK_EXT_attachment_feedback_loop_layout
    info:    attachmentFeedbackLoopLayout           : 1
    info:  VK_EXT_conservative_rasterization
    info:    extension supported                    : 1
    info:  VK_EXT_custom_border_color
    info:    customBorderColors                     : 1
    info:    customBorderColorWithoutFormat         : 1
    info:  VK_EXT_depth_clip_enable
    info:    depthClipEnable                        : 1
    info:  VK_EXT_depth_bias_control
    info:    depthBiasControl                       : 0
    info:    leastRepresentableValueForceUnormRepresentation : 0
    info:    floatRepresentation                    : 0
    info:    depthBiasExact                         : 0
    info:  VK_EXT_extended_dynamic_state3
    info:    extDynamicState3AlphaToCoverageEnable  : 1
    info:    extDynamicState3DepthClipEnable        : 1
    info:    extDynamicState3RasterizationSamples   : 1
    info:    extDynamicState3SampleMask             : 1
    info:    extDynamicState3LineRasterizationMode  : 1
    info:  VK_EXT_fragment_shader_interlock
    info:    fragmentShaderSampleInterlock          : 0
    info:    fragmentShaderPixelInterlock           : 0
    info:  VK_EXT_full_screen_exclusive
    info:    extension supported                    : 0
    info:  VK_EXT_graphics_pipeline_library
    info:    graphicsPipelineLibrary                : 1
    info:  VK_EXT_line_rasterization
    info:    rectangularLines                       : 1
    info:    smoothLines                            : 1
    info:  VK_EXT_memory_budget
    info:    extension supported                    : 1
    info:  VK_EXT_memory_priority
    info:    memoryPriority                         : 1
    info:  VK_EXT_non_seamless_cube_map
    info:    nonSeamlessCubeMap                     : 1
    info:  VK_EXT_robustness2
    info:    robustBufferAccess2                    : 1
    info:    robustImageAccess2                     : 1
    info:    nullDescriptor                         : 1
    info:  VK_EXT_shader_module_identifier
    info:    shaderModuleIdentifier                 : 1
    info:  VK_EXT_shader_stencil_export
    info:    extension supported                    : 0
    info:  VK_EXT_swapchain_colorspace
    info:    extension supported                    : 0
    info:  VK_EXT_swapchain_maintenance1
    info:    swapchainMaintenance1                  : 0
    info:  VK_EXT_hdr_metadata
    info:    extension supported                    : 0
    info:  VK_EXT_transform_feedback
    info:    transformFeedback                      : 0
    info:    geometryStreams                        : 0
    info:  VK_EXT_vertex_attribute_divisor
    info:    vertexAttributeInstanceRateDivisor     : 1
    info:    vertexAttributeInstanceRateZeroDivisor : 1
    info:  VK_KHR_external_memory_win32
    info:    extension supported                    : 0
    info:  VK_KHR_external_semaphore_win32
    info:    extension supported                    : 0
    info:  VK_KHR_maintenance5
    info:    maintenance5                           : 0
    info:  VK_KHR_present_id
    info:    presentId                              : 1
    info:  VK_KHR_present_wait
    info:    presentWait                            : 1
    info:  VK_NVX_binary_import
    info:    extension supported                    : 0
    info:  VK_NVX_image_view_handle
    info:    extension supported                    : 0
    info:  VK_KHR_win32_keyed_mutex
    info:    extension supported                    : 0
    info:  Queue families:
    info:    Graphics : 0
    info:    Transfer : 1
    info:    Sparse   : 0
    info:  DXVK: Read 149 valid state cache entries
    info:  DXVK: Graphics pipeline libraries supported
    info:  D3D9DeviceEx: Using extended constant set for software vertex processing.
    info:  D3D9DeviceEx::ResetSwapChain:
    info:    Requested Presentation Parameters
    info:      - Width:              0
    info:      - Height:             0
    info:      - Format:             D3D9Format::A8R8G8B8
    info:      - Auto Depth Stencil: true
    info:                  ^ Format: D3D9Format::D24S8
    info:      - Windowed:           true
    info:      - Swap effect:        1
    info:  Presenter: Actual swap chain properties:
    info:    Format:       VK_FORMAT_B8G8R8A8_UNORM
    info:    Color space:  VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
    info:    Present mode: VK_PRESENT_MODE_IMMEDIATE_KHR (dynamic: no)
    info:    Buffer size:  640x480
    info:    Image count:  3
    info:    Exclusive FS: 0
    warn:  ConvertFormat: Unknown format encountered: 1397249614
    warn:  ConvertFormat: Unknown format encountered: 1397249614
    warn:  ConvertFormat: Unknown format encountered: 1397249614
    warn:  ConvertFormat: Unknown format encountered: 1397249614
    warn:  ConvertFormat: Unknown format encountered: 1397249614
    info:  DXVK: Using 16 compiler threads
    info:  Presenter: Actual swap chain properties:
    info:    Format:       VK_FORMAT_B8G8R8A8_UNORM
    info:    Color space:  VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
    info:    Present mode: VK_PRESENT_MODE_IMMEDIATE_KHR (dynamic: no)
    info:    Buffer size:  640x480
    info:    Image count:  3
    info:    Exclusive FS: 0
    01c4:fixme:kernelbase:AppPolicyGetProcessTerminationMethod FFFFFFFA, 0051FEA8
    027c:fixme:advapi:GetCurrentHwProfileA (0BDCF9B8) semi-stub
    Found Trackmania process id: self.tm_process_id=134712
    Initialize connection to TMInterface
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    ERROR: ld.so: object 'libgamemodeauto.so.0' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    ERROR: ld.so: object 'libgamemodeauto.so.0' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    Connected
    ERROR: ld.so: object 'libgamemodeauto.so.0' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.
    ERROR: ld.so: object 'libgamemodeauto.so.0' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.
    02ac:fixme:ntdll:NtQuerySystemInformation info_class SYSTEM_PERFORMANCE_INFORMATION
    All rollout queues were empty. Learner sleeps 1 second.
    Requested map load
    02b8:fixme:kernelbase:AppPolicyGetThreadInitializationType FFFFFFFA, 1591FF08
    info:  Game: TmForever.exe
    info:  DXVK: v2.3
    info:  Vulkan: Found vkGetInstanceProcAddr in winevulkan.dll @ 0x75b23760
    info:  Built-in extension providers:
    info:    Win32 WSI
    info:    OpenVR
    info:    OpenXR
    info:  OpenVR: could not open registry key, status 2
    info:  OpenVR: Failed to locate module
    info:  Enabled instance extensions:
    info:    VK_KHR_get_surface_capabilities2
    info:    VK_KHR_surface
    info:    VK_KHR_win32_surface
    warn:  Skipping CPU adapter: llvmpipe (LLVM 15.0.7, 256 bits)
    info:  D3D9: VK_FORMAT_D16_UNORM_S8_UINT -> VK_FORMAT_D24_UNORM_S8_UINT
    info:  NVIDIA GeForce RTX 4070 Ti:
    info:    Driver : NVIDIA 535.129.3
    info:    Memory Heap[0]:
    info:      Size: 12282 MiB
    info:      Flags: 0x1
    info:      Memory Type[1]: Property Flags = 0x1
    info:    Memory Heap[1]:
    info:      Size: 48130 MiB
    info:      Flags: 0x0
    info:      Memory Type[0]: Property Flags = 0x0
    info:      Memory Type[2]: Property Flags = 0x6
    info:      Memory Type[3]: Property Flags = 0xe
    info:    Memory Heap[2]:
    info:      Size: 246 MiB
    info:      Flags: 0x1
    info:      Memory Type[4]: Property Flags = 0x7
    info:  Process set as DPI aware
    info:  Game: TmForever.exe
    info:  DXVK: v2.3
    info:  Vulkan: Found vkGetInstanceProcAddr in winevulkan.dll @ 0x75b23760
    info:  Built-in extension providers:
    info:    Win32 WSI
    info:    OpenVR
    info:    OpenXR
    info:  OpenVR: could not open registry key, status 2
    info:  OpenVR: Failed to locate module
    info:  Enabled instance extensions:
    info:    VK_KHR_get_surface_capabilities2
    info:    VK_KHR_surface
    info:    VK_KHR_win32_surface
    warn:  Skipping CPU adapter: llvmpipe (LLVM 15.0.7, 256 bits)
    info:  D3D9: VK_FORMAT_D16_UNORM_S8_UINT -> VK_FORMAT_D24_UNORM_S8_UINT
    info:  NVIDIA GeForce RTX 4070 Ti:
    info:    Driver : NVIDIA 535.129.3
    info:    Memory Heap[0]:
    info:      Size: 12282 MiB
    info:      Flags: 0x1
    info:      Memory Type[1]: Property Flags = 0x1
    info:    Memory Heap[1]:
    info:      Size: 48130 MiB
    info:      Flags: 0x0
    info:      Memory Type[0]: Property Flags = 0x0
    info:      Memory Type[2]: Property Flags = 0x6
    info:      Memory Type[3]: Property Flags = 0xe
    info:    Memory Heap[2]:
    info:      Size: 246 MiB
    info:      Flags: 0x1
    info:      Memory Type[4]: Property Flags = 0x7
    info:  Process set as DPI aware
    info:  Device properties:
    info:    Device : NVIDIA GeForce RTX 4070 Ti
    info:    Driver : NVIDIA 535.129.3
    info:  Enabled device extensions:
    info:    VK_EXT_attachment_feedback_loop_layout
    info:    VK_EXT_conservative_rasterization
    info:    VK_EXT_custom_border_color
    info:    VK_EXT_depth_clip_enable
    info:    VK_EXT_extended_dynamic_state3
    info:    VK_EXT_fragment_shader_interlock
    info:    VK_EXT_graphics_pipeline_library
    info:    VK_EXT_memory_priority
    info:    VK_EXT_non_seamless_cube_map
    info:    VK_EXT_robustness2
    info:    VK_EXT_shader_module_identifier
    info:    VK_EXT_transform_feedback
    info:    VK_EXT_vertex_attribute_divisor
    info:    VK_KHR_pipeline_library
    info:    VK_KHR_present_id
    info:    VK_KHR_present_wait
    info:    VK_KHR_swapchain
    info:  Device features:
    info:    robustBufferAccess                     : 1
    info:    fullDrawIndexUint32                    : 1
    info:    imageCubeArray                         : 1
    info:    independentBlend                       : 1
    info:    geometryShader                         : 1
    info:    tessellationShader                     : 0
    info:    sampleRateShading                      : 1
    info:    dualSrcBlend                           : 0
    info:    logicOp                                : 0
    info:    multiDrawIndirect                      : 0
    info:    drawIndirectFirstInstance              : 0
    info:    depthClamp                             : 1
    info:    depthBiasClamp                         : 1
    info:    fillModeNonSolid                       : 1
    info:    depthBounds                            : 1
    info:    wideLines                              : 1
    info:    multiViewport                          : 1
    info:    samplerAnisotropy                      : 1
    info:    textureCompressionBC                   : 1
    info:    occlusionQueryPrecise                  : 1
    info:    pipelineStatisticsQuery                : 1
    info:    vertexPipelineStoresAndAtomics         : 1
    info:    fragmentStoresAndAtomics               : 0
    info:    shaderImageGatherExtended              : 0
    info:    shaderClipDistance                     : 1
    info:    shaderCullDistance                     : 1
    info:    shaderFloat64                          : 0
    info:    shaderInt64                            : 0
    info:    variableMultisampleRate                : 1
    info:    shaderResourceResidency                : 0
    info:    shaderResourceMinLod                   : 0
    info:    sparseBinding                          : 0
    info:    sparseResidencyBuffer                  : 0
    info:    sparseResidencyImage2D                 : 0
    info:    sparseResidencyImage3D                 : 0
    info:    sparseResidency2Samples                : 0
    info:    sparseResidency4Samples                : 0
    info:    sparseResidency8Samples                : 0
    info:    sparseResidency16Samples               : 0
    info:    sparseResidencyAliased                 : 0
    info:  Vulkan 1.1
    info:    shaderDrawParameters                   : 0
    info:  Vulkan 1.2
    info:    samplerMirrorClampToEdge               : 1
    info:    drawIndirectCount                      : 1
    info:    samplerFilterMinmax                    : 0
    info:    hostQueryReset                         : 1
    info:    timelineSemaphore                      : 1
    info:    bufferDeviceAddress                    : 0
    info:    shaderOutputViewportIndex              : 1
    info:    shaderOutputLayer                      : 1
    info:    vulkanMemoryModel                      : 1
    info:  Vulkan 1.3
    info:    robustImageAccess                      : 0
    info:    pipelineCreationCacheControl           : 1
    info:    shaderDemoteToHelperInvocation         : 1
    info:    shaderZeroInitializeWorkgroupMemory    : 0
    info:    synchronization2                       : 1
    info:    dynamicRendering                       : 1
    info:  VK_AMD_shader_fragment_mask
    info:    extension supported                    : 0
    info:  VK_EXT_attachment_feedback_loop_layout
    info:    attachmentFeedbackLoopLayout           : 1
    info:  VK_EXT_conservative_rasterization
    info:    extension supported                    : 1
    info:  VK_EXT_custom_border_color
    info:    customBorderColors                     : 1
    info:    customBorderColorWithoutFormat         : 1
    info:  VK_EXT_depth_clip_enable
    info:    depthClipEnable                        : 1
    info:  VK_EXT_depth_bias_control
    info:    depthBiasControl                       : 0
    info:    leastRepresentableValueForceUnormRepresentation : 0
    info:    floatRepresentation                    : 0
    info:    depthBiasExact                         : 0
    info:  VK_EXT_extended_dynamic_state3
    info:    extDynamicState3AlphaToCoverageEnable  : 1
    info:    extDynamicState3DepthClipEnable        : 1
    info:    extDynamicState3RasterizationSamples   : 1
    info:    extDynamicState3SampleMask             : 1
    info:    extDynamicState3LineRasterizationMode  : 1
    info:  VK_EXT_fragment_shader_interlock
    info:    fragmentShaderSampleInterlock          : 0
    info:    fragmentShaderPixelInterlock           : 0
    info:  VK_EXT_full_screen_exclusive
    info:    extension supported                    : 0
    info:  VK_EXT_graphics_pipeline_library
    info:    graphicsPipelineLibrary                : 1
    info:  VK_EXT_line_rasterization
    info:    rectangularLines                       : 1
    info:    smoothLines                            : 1
    info:  VK_EXT_memory_budget
    info:    extension supported                    : 1
    info:  VK_EXT_memory_priority
    info:    memoryPriority                         : 1
    info:  VK_EXT_non_seamless_cube_map
    info:    nonSeamlessCubeMap                     : 1
    info:  VK_EXT_robustness2
    info:    robustBufferAccess2                    : 1
    info:    robustImageAccess2                     : 1
    info:    nullDescriptor                         : 1
    info:  VK_EXT_shader_module_identifier
    info:    shaderModuleIdentifier                 : 1
    info:  VK_EXT_shader_stencil_export
    info:    extension supported                    : 0
    info:  VK_EXT_swapchain_colorspace
    info:    extension supported                    : 0
    info:  VK_EXT_swapchain_maintenance1
    info:    swapchainMaintenance1                  : 0
    info:  VK_EXT_hdr_metadata
    info:    extension supported                    : 0
    info:  VK_EXT_transform_feedback
    info:    transformFeedback                      : 0
    info:    geometryStreams                        : 0
    info:  VK_EXT_vertex_attribute_divisor
    info:    vertexAttributeInstanceRateDivisor     : 1
    info:    vertexAttributeInstanceRateZeroDivisor : 1
    info:  VK_KHR_external_memory_win32
    info:    extension supported                    : 0
    info:  VK_KHR_external_semaphore_win32
    info:    extension supported                    : 0
    info:  VK_KHR_maintenance5
    info:    maintenance5                           : 0
    info:  VK_KHR_present_id
    info:    presentId                              : 1
    info:  VK_KHR_present_wait
    info:    presentWait                            : 1
    info:  VK_NVX_binary_import
    info:    extension supported                    : 0
    info:  VK_NVX_image_view_handle
    info:    extension supported                    : 0
    info:  VK_KHR_win32_keyed_mutex
    info:    extension supported                    : 0
    info:  Queue families:
    info:    Graphics : 0
    info:    Transfer : 1
    info:    Sparse   : 0
    info:  DXVK: Read 149 valid state cache entries
    info:  DXVK: Graphics pipeline libraries supported
    info:  D3D9DeviceEx: Using extended constant set for software vertex processing.
    info:  D3D9DeviceEx::ResetSwapChain:
    info:    Requested Presentation Parameters
    info:      - Width:              0
    info:      - Height:             0
    info:      - Format:             D3D9Format::A8R8G8B8
    info:      - Auto Depth Stencil: true
    info:                  ^ Format: D3D9Format::D24S8
    info:      - Windowed:           true
    info:      - Swap effect:        1
    info:  Presenter: Actual swap chain properties:
    info:    Format:       VK_FORMAT_B8G8R8A8_UNORM
    info:    Color space:  VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
    info:    Present mode: VK_PRESENT_MODE_IMMEDIATE_KHR (dynamic: no)
    info:    Buffer size:  640x480
    info:    Image count:  3
    info:    Exclusive FS: 0
    warn:  ConvertFormat: Unknown format encountered: 1397249614
    warn:  ConvertFormat: Unknown format encountered: 1397249614
    warn:  ConvertFormat: Unknown format encountered: 1397249614
    warn:  ConvertFormat: Unknown format encountered: 1397249614
    warn:  ConvertFormat: Unknown format encountered: 1397249614
    info:  DXVK: Using 16 compiler threads
    info:  Presenter: Actual swap chain properties:
    info:    Format:       VK_FORMAT_B8G8R8A8_UNORM
    info:    Color space:  VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
    info:    Present mode: VK_PRESENT_MODE_IMMEDIATE_KHR (dynamic: no)
    info:    Buffer size:  640x480
    info:    Image count:  3
    info:    Exclusive FS: 0
    0294:fixme:kernelbase:AppPolicyGetProcessTerminationMethod FFFFFFFA, 0051FEA8
    034c:fixme:advapi:GetCurrentHwProfileA (0BDCF9B8) semi-stub
    Found Trackmania process id: self.tm_process_id=134771
    All rollout queues were empty. Learner sleeps 1 second.
    Initialize connection to TMInterface
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    ERROR: ld.so: object 'libgamemodeauto.so.0' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    ERROR: ld.so: object 'libgamemodeauto.so.0' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    Connected
    ERROR: ld.so: object 'libgamemodeauto.so.0' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.
    ERROR: ld.so: object 'libgamemodeauto.so.0' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.
    037c:fixme:ntdll:NtQuerySystemInformation info_class SYSTEM_PERFORMANCE_INFORMATION
    Requested map load
    0388:fixme:kernelbase:AppPolicyGetThreadInitializationType FFFFFFFA, 1591FF08
    info:  Game: TmForever.exe
    info:  DXVK: v2.3
    info:  Vulkan: Found vkGetInstanceProcAddr in winevulkan.dll @ 0x75b23760
    info:  Built-in extension providers:
    info:    Win32 WSI
    info:    OpenVR
    info:    OpenXR
    info:  OpenVR: could not open registry key, status 2
    info:  OpenVR: Failed to locate module
    info:  Enabled instance extensions:
    info:    VK_KHR_get_surface_capabilities2
    info:    VK_KHR_surface
    info:    VK_KHR_win32_surface
    warn:  Skipping CPU adapter: llvmpipe (LLVM 15.0.7, 256 bits)
    info:  D3D9: VK_FORMAT_D16_UNORM_S8_UINT -> VK_FORMAT_D24_UNORM_S8_UINT
    info:  NVIDIA GeForce RTX 4070 Ti:
    info:    Driver : NVIDIA 535.129.3
    info:    Memory Heap[0]:
    info:      Size: 12282 MiB
    info:      Flags: 0x1
    info:      Memory Type[1]: Property Flags = 0x1
    info:    Memory Heap[1]:
    info:      Size: 48130 MiB
    info:      Flags: 0x0
    info:      Memory Type[0]: Property Flags = 0x0
    info:      Memory Type[2]: Property Flags = 0x6
    info:      Memory Type[3]: Property Flags = 0xe
    info:    Memory Heap[2]:
    info:      Size: 246 MiB
    info:      Flags: 0x1
    info:      Memory Type[4]: Property Flags = 0x7
    info:  Process set as DPI aware
    info:  Game: TmForever.exe
    info:  DXVK: v2.3
    info:  Vulkan: Found vkGetInstanceProcAddr in winevulkan.dll @ 0x75b23760
    info:  Built-in extension providers:
    info:    Win32 WSI
    info:    OpenVR
    info:    OpenXR
    info:  OpenVR: could not open registry key, status 2
    info:  OpenVR: Failed to locate module
    info:  Enabled instance extensions:
    info:    VK_KHR_get_surface_capabilities2
    info:    VK_KHR_surface
    info:    VK_KHR_win32_surface
    warn:  Skipping CPU adapter: llvmpipe (LLVM 15.0.7, 256 bits)
    info:  D3D9: VK_FORMAT_D16_UNORM_S8_UINT -> VK_FORMAT_D24_UNORM_S8_UINT
    info:  NVIDIA GeForce RTX 4070 Ti:
    info:    Driver : NVIDIA 535.129.3
    info:    Memory Heap[0]:
    info:      Size: 12282 MiB
    info:      Flags: 0x1
    info:      Memory Type[1]: Property Flags = 0x1
    info:    Memory Heap[1]:
    info:      Size: 48130 MiB
    info:      Flags: 0x0
    info:      Memory Type[0]: Property Flags = 0x0
    info:      Memory Type[2]: Property Flags = 0x6
    info:      Memory Type[3]: Property Flags = 0xe
    info:    Memory Heap[2]:
    info:      Size: 246 MiB
    info:      Flags: 0x1
    info:      Memory Type[4]: Property Flags = 0x7
    info:  Process set as DPI aware
    info:  Device properties:
    info:    Device : NVIDIA GeForce RTX 4070 Ti
    info:    Driver : NVIDIA 535.129.3
    info:  Enabled device extensions:
    info:    VK_EXT_attachment_feedback_loop_layout
    info:    VK_EXT_conservative_rasterization
    info:    VK_EXT_custom_border_color
    info:    VK_EXT_depth_clip_enable
    info:    VK_EXT_extended_dynamic_state3
    info:    VK_EXT_fragment_shader_interlock
    info:    VK_EXT_graphics_pipeline_library
    info:    VK_EXT_memory_priority
    info:    VK_EXT_non_seamless_cube_map
    info:    VK_EXT_robustness2
    info:    VK_EXT_shader_module_identifier
    info:    VK_EXT_transform_feedback
    info:    VK_EXT_vertex_attribute_divisor
    info:    VK_KHR_pipeline_library
    info:    VK_KHR_present_id
    info:    VK_KHR_present_wait
    info:    VK_KHR_swapchain
    info:  Device features:
    info:    robustBufferAccess                     : 1
    info:    fullDrawIndexUint32                    : 1
    info:    imageCubeArray                         : 1
    info:    independentBlend                       : 1
    info:    geometryShader                         : 1
    info:    tessellationShader                     : 0
    info:    sampleRateShading                      : 1
    info:    dualSrcBlend                           : 0
    info:    logicOp                                : 0
    info:    multiDrawIndirect                      : 0
    info:    drawIndirectFirstInstance              : 0
    info:    depthClamp                             : 1
    info:    depthBiasClamp                         : 1
    info:    fillModeNonSolid                       : 1
    info:    depthBounds                            : 1
    info:    wideLines                              : 1
    info:    multiViewport                          : 1
    info:    samplerAnisotropy                      : 1
    info:    textureCompressionBC                   : 1
    info:    occlusionQueryPrecise                  : 1
    info:    pipelineStatisticsQuery                : 1
    info:    vertexPipelineStoresAndAtomics         : 1
    info:    fragmentStoresAndAtomics               : 0
    info:    shaderImageGatherExtended              : 0
    info:    shaderClipDistance                     : 1
    info:    shaderCullDistance                     : 1
    info:    shaderFloat64                          : 0
    info:    shaderInt64                            : 0
    info:    variableMultisampleRate                : 1
    info:    shaderResourceResidency                : 0
    info:    shaderResourceMinLod                   : 0
    info:    sparseBinding                          : 0
    info:    sparseResidencyBuffer                  : 0
    info:    sparseResidencyImage2D                 : 0
    info:    sparseResidencyImage3D                 : 0
    info:    sparseResidency2Samples                : 0
    info:    sparseResidency4Samples                : 0
    info:    sparseResidency8Samples                : 0
    info:    sparseResidency16Samples               : 0
    info:    sparseResidencyAliased                 : 0
    info:  Vulkan 1.1
    info:    shaderDrawParameters                   : 0
    info:  Vulkan 1.2
    info:    samplerMirrorClampToEdge               : 1
    info:    drawIndirectCount                      : 1
    info:    samplerFilterMinmax                    : 0
    info:    hostQueryReset                         : 1
    info:    timelineSemaphore                      : 1
    info:    bufferDeviceAddress                    : 0
    info:    shaderOutputViewportIndex              : 1
    info:    shaderOutputLayer                      : 1
    info:    vulkanMemoryModel                      : 1
    info:  Vulkan 1.3
    info:    robustImageAccess                      : 0
    info:    pipelineCreationCacheControl           : 1
    info:    shaderDemoteToHelperInvocation         : 1
    info:    shaderZeroInitializeWorkgroupMemory    : 0
    info:    synchronization2                       : 1
    info:    dynamicRendering                       : 1
    info:  VK_AMD_shader_fragment_mask
    info:    extension supported                    : 0
    info:  VK_EXT_attachment_feedback_loop_layout
    info:    attachmentFeedbackLoopLayout           : 1
    info:  VK_EXT_conservative_rasterization
    info:    extension supported                    : 1
    info:  VK_EXT_custom_border_color
    info:    customBorderColors                     : 1
    info:    customBorderColorWithoutFormat         : 1
    info:  VK_EXT_depth_clip_enable
    info:    depthClipEnable                        : 1
    info:  VK_EXT_depth_bias_control
    info:    depthBiasControl                       : 0
    info:    leastRepresentableValueForceUnormRepresentation : 0
    info:    floatRepresentation                    : 0
    info:    depthBiasExact                         : 0
    info:  VK_EXT_extended_dynamic_state3
    info:    extDynamicState3AlphaToCoverageEnable  : 1
    info:    extDynamicState3DepthClipEnable        : 1
    info:    extDynamicState3RasterizationSamples   : 1
    info:    extDynamicState3SampleMask             : 1
    info:    extDynamicState3LineRasterizationMode  : 1
    info:  VK_EXT_fragment_shader_interlock
    info:    fragmentShaderSampleInterlock          : 0
    info:    fragmentShaderPixelInterlock           : 0
    info:  VK_EXT_full_screen_exclusive
    info:    extension supported                    : 0
    info:  VK_EXT_graphics_pipeline_library
    info:    graphicsPipelineLibrary                : 1
    info:  VK_EXT_line_rasterization
    info:    rectangularLines                       : 1
    info:    smoothLines                            : 1
    info:  VK_EXT_memory_budget
    info:    extension supported                    : 1
    info:  VK_EXT_memory_priority
    info:    memoryPriority                         : 1
    info:  VK_EXT_non_seamless_cube_map
    info:    nonSeamlessCubeMap                     : 1
    info:  VK_EXT_robustness2
    info:    robustBufferAccess2                    : 1
    info:    robustImageAccess2                     : 1
    info:    nullDescriptor                         : 1
    info:  VK_EXT_shader_module_identifier
    info:    shaderModuleIdentifier                 : 1
    info:  VK_EXT_shader_stencil_export
    info:    extension supported                    : 0
    info:  VK_EXT_swapchain_colorspace
    info:    extension supported                    : 0
    info:  VK_EXT_swapchain_maintenance1
    info:    swapchainMaintenance1                  : 0
    info:  VK_EXT_hdr_metadata
    info:    extension supported                    : 0
    info:  VK_EXT_transform_feedback
    info:    transformFeedback                      : 0
    info:    geometryStreams                        : 0
    info:  VK_EXT_vertex_attribute_divisor
    info:    vertexAttributeInstanceRateDivisor     : 1
    info:    vertexAttributeInstanceRateZeroDivisor : 1
    info:  VK_KHR_external_memory_win32
    info:    extension supported                    : 0
    info:  VK_KHR_external_semaphore_win32
    info:    extension supported                    : 0
    info:  VK_KHR_maintenance5
    info:    maintenance5                           : 0
    info:  VK_KHR_present_id
    info:    presentId                              : 1
    info:  VK_KHR_present_wait
    info:    presentWait                            : 1
    info:  VK_NVX_binary_import
    info:    extension supported                    : 0
    info:  VK_NVX_image_view_handle
    info:    extension supported                    : 0
    info:  VK_KHR_win32_keyed_mutex
    info:    extension supported                    : 0
    info:  Queue families:
    info:    Graphics : 0
    info:    Transfer : 1
    info:    Sparse   : 0
    info:  DXVK: Read 149 valid state cache entries
    info:  DXVK: Graphics pipeline libraries supported
    info:  D3D9DeviceEx: Using extended constant set for software vertex processing.
    info:  D3D9DeviceEx::ResetSwapChain:
    info:    Requested Presentation Parameters
    info:      - Width:              0
    info:      - Height:             0
    info:      - Format:             D3D9Format::A8R8G8B8
    info:      - Auto Depth Stencil: true
    info:                  ^ Format: D3D9Format::D24S8
    info:      - Windowed:           true
    info:      - Swap effect:        1
    info:  Presenter: Actual swap chain properties:
    info:    Format:       VK_FORMAT_B8G8R8A8_UNORM
    info:    Color space:  VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
    info:    Present mode: VK_PRESENT_MODE_IMMEDIATE_KHR (dynamic: no)
    info:    Buffer size:  640x480
    info:    Image count:  3
    info:    Exclusive FS: 0
    All rollout queues were empty. Learner sleeps 1 second.
    warn:  ConvertFormat: Unknown format encountered: 1397249614
    warn:  ConvertFormat: Unknown format encountered: 1397249614
    warn:  ConvertFormat: Unknown format encountered: 1397249614
    warn:  ConvertFormat: Unknown format encountered: 1397249614
    warn:  ConvertFormat: Unknown format encountered: 1397249614
    info:  DXVK: Using 16 compiler threads
    info:  Presenter: Actual swap chain properties:
    info:    Format:       VK_FORMAT_B8G8R8A8_UNORM
    info:    Color space:  VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
    info:    Present mode: VK_PRESENT_MODE_IMMEDIATE_KHR (dynamic: no)
    info:    Buffer size:  640x480
    info:    Image count:  3
    info:    Exclusive FS: 0
    0364:fixme:kernelbase:AppPolicyGetProcessTerminationMethod FFFFFFFA, 0051FEA8
    041c:fixme:advapi:GetCurrentHwProfileA (0BDCF9B8) semi-stub
    Found Trackmania process id: self.tm_process_id=134830
    Initialize connection to TMInterface
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    [Errno 111] Connection refused
    Connected
    Requested map load
    0438:fixme:kernelbase:AppPolicyGetThreadInitializationType FFFFFFFA, 1591FF08
    All rollout queues were empty. Learner sleeps 1 second.

    Race time ratio   3.075628773050643
     NMG=474

    All rollout queues were empty. Learner sleeps 1 second.
    All rollout queues were empty. Learner sleeps 1 second.

    Race time ratio   2.624520098598898
     NMG=948

    All rollout queues were empty. Learner sleeps 1 second.
