for seed in 1234 2345 3456
do
python3 main.py -m model.loss.backprop=True,False model.loss.loss_sup=pred,predsim train.seed=$seed
done