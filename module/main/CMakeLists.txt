idf_component_register (
    SRCS "main.c"
    SRCS "../i2s/i2s_input.c"
    SRCS "../sd_card/sdcard.c"

    INCLUDE_DIRS "."
    INCLUDE_DIRS "../i2s"
    INCLUDE_DIRS "../sd_card"
)


component_compile_options(-w)
target_compile_options(${COMPONENT_LIB} PRIVATE "-Wno-format")
