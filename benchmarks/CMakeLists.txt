set(SUBDIRECTORIES
  test_fwd_conv
  test_bwd_conv
  test_fwd_pool
  test_bwd_pool
  test_fwd_lrn
  test_bwd_lrn
  test_fwd_activation
  test_bwd_activation
  test_fwd_fc
  test_bwd_fc
  test_fwd_softmax
  test_bwd_softmax
  test_composed_model
)       

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB COMMON_FILE RELATIVE
     ${CMAKE_CURRENT_SOURCE_DIR}
     *.cc)
get_filename_component(COMMON_SRC ${COMMON_FILE} ABSOLUTE)
 
foreach(subdir ${SUBDIRECTORIES})
  add_subdirectory(${subdir})
endforeach(subdir)
