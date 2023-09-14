rm ./model/cora_best.pth 
mv ./model/cora.pth ./model/cora_best.pth 
rm ./pretrain_result/cora_best.npy
cp ./pretrain_result/cora.npy ./pretrain_result/cora_best.npy