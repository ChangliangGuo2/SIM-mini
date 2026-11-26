function  [output1, output2, output3] = second_RL(data_fI2, order, fidelity, fidelity_z, sparsity, Ipsf2, finter2, iter2, backg2)
data_fI2 = abs(single(fourierInterpolation(data_fI2.^(1/order),[finter2,finter2,finter2],'both')));
output1 = data_fI2./max(max(data_fI2(:)));

[data_fI2] = sparse_main(abs(data_fI2),fidelity,fidelity_z,sparsity,100,backg2);
output2 = data_fI2./max(max(data_fI2(:)));

data_fI2 = double(data_fI2).^order;
data_fI2 = gather(RL3D(data_fI2,Ipsf2.^order,iter2,1));
data_fI2 = abs(data_fI2.^(1/order));
output3 = percennorm(data_fI2, 0.5, 99.99);
end