Traceback (most recent call last):
  File "inverse_PINN_AD_trueY_general.py", line 587, in <module>
    run_ad_truth(opt)
  File "inverse_PINN_AD_trueY_general.py", line 560, in run_ad_truth
    model = train_ad(model, args, config, now_string)
  File "inverse_PINN_AD_trueY_general.py", line 368, in train_ad
    loss, loss_list, _ = model.loss()
  File "inverse_PINN_AD_trueY_general.py", line 300, in loss
    all_loss4 += loss_4
UnboundLocalError: local variable 'all_loss4' referenced before assignment
