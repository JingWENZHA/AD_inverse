import matplotlib.pyplot as plt
import scipy.io
from utils import *

# self.true_para = torch.tensor([self.k_a, self.k_ta, self.k_mt, self.d_a, self.theta,
#                                                self.k_t, self.k_at, self.k_ma, self.d_t, self.delta,
#                                                self.k_r, self.k_tn, self.k_mtn, self.gamma, self.k_an,
#                                                self.k_man, self.beta, self.k_atn])
# para_matlab = [1.0000e-04, 1.0000e-03, 2.4540e-01, 6.0000e-02, 2.0000e+00,
#                3.9000e-03, 1.1000e-03, 1.0000e-03, 6.0000e-02, 2.0000e+00,
#                5.0000e-04, 2.3000e-03, 1.0000e+00, 2.0000e+00, 1.2000e-03,
#                8.2400e-01, 2.0000e+00, 1.2000e-03]
#
# para_inv_general = [4.9625659e-01, 7.6822180e-01, 8.8477433e-02, 5.9999999e-02, 2.0000000e+00,
#                     3.0742282e-01, 6.3407868e-01, 4.9009341e-01, 5.9999999e-02, 2.0000000e+00,
#                     5.0000002e-04, 4.5562798e-01, 4.0171731e-01, 2.0000000e+00, 6.3230628e-01,
#                     2.2325754e-02, 2.0000000e+00, 3.4889346e-01]

# no hill coefficient
para_matlab = [1.0000e-04, 1.0000e-03, 2.4540e-01, 6.0000e-02,
               3.9000e-03, 1.1000e-03, 1.0000e-03, 6.0000e-02,
               5.0000e-04, 2.3000e-03, 1.0000e+00, 1.2000e-03,
               8.2400e-01, 1.2000e-03]

para_inv_general = [4.9625659e-01, 7.6822180e-01, 8.8477433e-02, 5.9999999e-02,
                    3.0742282e-01, 6.3407868e-01, 4.9009341e-01, 5.9999999e-02,
                    5.0000002e-04, 4.5562798e-01, 4.0171731e-01,  6.3230628e-01,
                    2.2325754e-02, 3.4889346e-01]

plt.figure(figsize=(8, 6))

for i in range(14):
    # print(y_true_month[i]/0.1)

    plt.scatter(x = i, y = para_matlab[i], color = 'g', marker ='o')
    plt.scatter(x = i, y = para_inv_general[i], color = 'b', marker = 'x')



plt.plot([i for i in range(14)], para_matlab,'g')
plt.plot([i for i in range(14)], para_inv_general, 'b')
plt.legend(["matlab", "nn_inverse"])
plt.show()
plt.close()

