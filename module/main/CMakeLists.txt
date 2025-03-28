idf_component_register (
    SRCS "main.c"
    SRCS "../i2s/i2s_input.c"
    SRCS "../sd_card/sdcard.c"
    SRCS "../audio_tools/audio_tools.c"
    SRCS "../network/network_stream.c"

    INCLUDE_DIRS "."
    INCLUDE_DIRS "../i2s"
    INCLUDE_DIRS "../sd_card"
    INCLUDE_DIRS "../audio_tools"
    INCLUDE_DIRS "../network"
)


component_compile_options(-w)
target_compile_options(${COMPONENT_LIB} PRIVATE "-Wno-format")
