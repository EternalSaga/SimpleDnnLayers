file(GLOB_RECURSE NonOptCLibSrc "*.h" "*.c")
add_library(NonOptCLib STATIC "${NonOptCLibSrc}")