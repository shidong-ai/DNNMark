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
)       
 
foreach(subdir ${SUBDIRECTORIES})
  add_subdirectory(${subdir})
endforeach(subdir)
