Traceback (most recent call last):
  File "inverse_PINN_AD.py", line 511, in <module>
    run_ad_truth(opt)
  File "inverse_PINN_AD.py", line 470, in run_ad_truth
    model = train_ad(model, args, config, now_string)
  File "inverse_PINN_AD.py", line 348, in train_ad
    test_ad(model, args, config, now_string,param_ls, param_true, True, model.gt, None)
  File "inverse_PINN_AD.py", line 433, in test_ad
    m.add_subplot(x_list=[j for j in range(param_ls.shape[0])], y_lists = [param_ls[:,i], param_true[:,i]], color_list = 'b',fig_title=labels[i],line_style_list=["solid","dashed"])
  File "/deac/csc/chenGrp/zhanj318/PINN_AD2022/PINN_AD_cluster/utils.py", line 333, in add_subplot
    assert len(y_lists) == len(line_style_list) == len(color_list), "number of lines should be fixed"
AssertionError: number of lines should be fixed
