# # Calculate gradient on net_hv
# net_hv_delta = copy.deepcopy(net)
# # zero the gradient
# for p_hv_delta, p_delta in zip(net_hv_delta.parameters(), net_delta.parameters()):
#     p_hv_delta.data += args.approx_delta * p_delta.data
# # get the output
# net_hv_delta.zero_grad()
# outputs_hv_delta = net_hv_delta.forward(inputs_hv)
# # define the loss for calculating Hessian-vector product
# loss_hv_delta = criterion(outputs_hv_delta, targets_hv)
# # compute the gradient
# grad_hv_delta = torch.autograd.grad(loss_hv_delta, net_hv_delta.parameters())
#
# # calculate Hessian-vector product
# net_aux = copy.deepcopy(net)
# for p_aux, p_grad_hv, p_grad_hv_delta in zip(net_aux.parameters(), grad_hv, grad_hv_delta):
#     p_aux.data = (p_grad_hv_delta.data - p_grad_hv.data) / args.approx_delta
#
# # net_delta generate dummy grad
# net_delta_optimizer.zero_grad()
# net_delta_loss = criterion(net_delta.forward(inputs_hv[:10]), targets_hv[:10])
# net_delta_loss.backward()
#
# for p_delta_net, p_hv in zip(net_delta.parameters(), net_aux.parameters()):
#     p_delta_net.grad.data = p_hv.data * 1.0
#
# norm_net_delta_grad = 0.0
# for p in net_delta.parameters():
#     norm_net_delta_grad += p.grad.data.norm(2) ** 2
# norm_net_delta_grad = float(np.sqrt(norm_net_delta_grad))
# # print("norm_net_delta_grad, 1, before ", norm_net_delta_grad)


# # record the m(Delta) = g*Delta + Delta*H*Delta/2 + rho*\|Delta\|^3
# def m_delta():
#     m_1 = 0.0
#     for p_delta, p_grad_cubic in zip(net_delta.parameters(), grad_cubic):
#         m_1 += torch.sum(p_delta.data * p_grad_cubic.data)
#     m_2 = 0.0
#     for p_delta, p_hv_exact in zip(net_delta.parameters(), net_aux.parameters()):
#         m_2 += torch.sum(p_delta.data * p_hv_exact.data)
#     m_3 = 0.0
#     for p_delta in net_delta.parameters():
#         m_3 += p_delta.data.norm(2) ** 2
#     m_3 = torch.pow(torch.FloatTensor([m_3]), 1.5) * args.rho
#     m_delta_value = m_1 + m_2 + m_3[0]
#     return m_delta_value, (m_1, m_2, m_3[0])
# # print("epoch {}, m_delata: {}".format(epoch_c, m_delta()))
# if epoch_c == args.cubic_epoch - 1:
#     m_value, (m1, m2, m3) = m_delta()
#     m_deltas.append(m_value)
#     if np.abs(m_value) > 10.0:
#         print("{} {} ({}, {}, {})".format(np.array(m_deltas).mean(), m_value, m1, m2, m3))

# for a, b in zip(net_aux.parameters(), hv_exact):
#     print((a - b).abs().max())


# norm_net_delta_grad = 0.0
# for p in net_delta.parameters():
#     norm_net_delta_grad += p.grad.data.norm(2) ** 2
# norm_net_delta_grad = float(np.sqrt(norm_net_delta_grad))
# # print("norm_net_delta_grad: ", norm_net_delta_grad)