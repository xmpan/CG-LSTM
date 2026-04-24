import torch
import torch.nn as nn
import math
import numpy
from scipy.spatial.transform import Rotation

def mse_loss_self(y_true,y_pred):
	#计算真实值和预测值之间的差异
	diff=torch.tensor(y_true)- torch.tensor(y_pred)
	
	#计算差值的平方
	sq_diff=torch.square(diff)
	
	#计算均方误差，即平方差的平均值
	#使用np.mean计算平均值，axis=0表示沿着第一个轴（通常是样本维度）计算
	loss=torch.mean(sq_diff,axis=0)
	
	return loss

# diff = min(abs(y_true - y_pred), 2*torch.pi - (y_true - y_pred))
def mse_loss_grad(y_true, y_pred):
    #计算真实值和预测值之间的差异
    diff = y_true-y_pred
    #计算差值的平方
    sq_diff = torch.square(diff)
    #计算均方误差，即平方差的平均值
    loss = torch.mean(sq_diff)
    return loss

def huber_loss_self(y_true, y_pred, delta=1.0):
    # 计算预测值与真实值差异
    diff = y_true - y_pred
    abs_diff = torch.abs(diff)

    # 将L1损失和L2损失结合
    quadratic = torch.minimum(abs_diff, torch.tensor(delta)) # 计算小于delta的部分
    linear = abs_diff - quadratic # 计算超出delta部分的绝对误差
    loss = 0.5 * quadratic**2 + delta*linear # 计算总的huber损失

    return loss.mean() # 返回平均损失

def mse_loss(pred, target):
    criterion = nn.MSELoss()
    mseloss = torch.empty(target.shape[0], 1)
    for ii in range(target.shape[0]):
        mseloss[ii] = criterion(pred[ii], target[ii])
        # logger.info('pred[{}]={},target[{}]={}, mseloss[{}]={}'.format(ii, pred[ii], ii, target[ii],ii, mseloss[ii]))
    loss = torch.sum(mseloss)
    return loss

def quaternion2eulerbytensor(quaternion, is_degree : bool = False):
    r = Rotation.from_quat(quaternion.cpu().detach().numpy())
    euler = r.as_euler('xyz', degrees=is_degree)
    return euler

def quaternion2euler(quaternion, is_degree : bool = False):
    r = Rotation.from_quat(quaternion.cpu().numpy())
    euler = r.as_euler('xyz', degrees=is_degree)
    return euler


# Evaluate the model on test data
def evaluate(model,test_loader,device,epoch,logger,num_layers, num_class, method, correct_threshold):
    model.eval()  # switch to evaluation mode

    with torch.no_grad():
        total = 0
        correct = 0
        correct_2nd = 0
        loss_total = 0
        thetaerr_max = 0
        thetaerr_min = 3
        thetaerr_avg = 0
        phierr_max = 0
        phierr_min = 3
        phierr_avg = 0
        best_accuracy = 95
        for index, (hrrp, hrrp_omp, label, quat, theta_phi, radlos, filename) in enumerate(test_loader):
            hrrp = hrrp.to(device)
            hrrp_omp = hrrp_omp.to(device)
            quat = quat.to(device)
            label = label.to(device)
            theta_phi = theta_phi.to(device)

            hrrp = hrrp.float()
            hrrp_omp = hrrp_omp.float()
            if method == 'R6D':
                cl_pred, theta_pred, phi_pred = model(hrrp)
            else:
                cl_pred, qt_pred = model(hrrp, hrrp_omp)

            if method == 'multi_class_2out':
                total += theta_phi.shape[0]
            else:
                total += quat.shape[0]
            
            mseloss = torch.empty(quat.shape[0], 1)
            thetaloss = torch.empty(quat.shape[0], 1)
            philoss = torch.empty(quat.shape[0], 1)
            if method == 'multi_class_2out' or method == 'multi_class_2out_with_radarLos':
                for ii in range(theta_phi.shape[0]):
                    thetaloss[ii] = mse_loss_grad(qt_pred[ii][0], theta_phi[ii][0])
                    philoss[ii] = mse_loss_grad(qt_pred[ii][1], theta_phi[ii][1])
                loss = torch.mean(thetaloss) + torch.mean(philoss)
                loss_total += loss
                for ii in range(theta_phi.shape[0]):
                    thetaerr = abs(qt_pred[ii][0] - theta_phi[ii][0])*180/math.pi
                    phierr = abs(qt_pred[ii][1] - theta_phi[ii][1])*180/math.pi
                    avg_err = sum([thetaerr, phierr])/2
                    if avg_err < correct_threshold:
                        correct += 1
                    if avg_err < correct_threshold*2:
                        correct_2nd += 1
            elif method == 'R6D':
                for ii in range(theta_phi.shape[0]):
                    thetaloss[ii] = mse_loss_grad(theta_pred[ii], theta_phi[ii][0])
                    philoss[ii] = mse_loss_grad(phi_pred[ii], theta_phi[ii][1])
                loss = torch.mean(thetaloss) + torch.mean(philoss)
                loss_total += loss
                for ii in range(theta_phi.shape[0]):
                    thetaerr = min(abs(theta_pred[ii]-theta_phi[ii][0]), 2*math.pi - abs(theta_pred[ii]-theta_phi[ii][0]))*180/math.pi
                    phierr = min(abs(phi_pred[ii]-theta_phi[ii][1]), 2*math.pi - abs(phi_pred[ii]-theta_phi[ii][1]))*180/math.pi
                    avg_err = sum([thetaerr, phierr])/2

                    if thetaerr > 3 and phierr > 3:
                        logger.info('thetaerr: {}, phierr: {}, file: {}'.format(thetaerr, phierr, filename[ii]))
                    if thetaerr_min > thetaerr:
                        thetaerr_min = thetaerr
                    if thetaerr_max < thetaerr:
                        thetaerr_max = thetaerr
                    thetaerr_avg += thetaerr

                    if phierr_min > phierr:
                        phierr_min = phierr
                    if phierr_max < phierr:
                        phierr_max = phierr
                    phierr_avg += phierr

                    if avg_err < correct_threshold:
                        correct += 1
                    if avg_err < correct_threshold*2:
                        correct_2nd += 1
            elif method == 'multi_class_3out':
                for ii in range(qt_pred.shape[0]):
                    qt_target = quaternion2eulerbytensor(quat[ii])
                    qtx = (qt_pred[ii][0] - qt_target[0])*180/torch.pi
                    qty = (qt_pred[ii][1] - qt_target[1])*180/torch.pi
                    qtz = (qt_pred[ii][2] - qt_target[2])*180/torch.pi
                    mseloss[ii] = sum([abs(qtx), abs(qty), abs(qtz)])/3
                    if mseloss[ii] < correct_threshold:
                        correct += 1
                    if mseloss[ii] < correct_threshold*2:
                        correct_2nd += 1
                    if mseloss[ii] > correct_threshold*10:
                        logger.info('label={}, qtx={:.4},qty={:.4},qtz={:.4}, filename={}'.format(label[ii]+1, qtx,qty,qtz, filename[ii]))
                        # with open('evaluate_error_case_10d3_test-7.txt', 'a') as f:
                        #     f.write(' '.join(map(str, [label[ii].item()+1, qtx.item(), qty.item(), qtz.item(), filename[ii]])) + '\n')
                qt_target = torch.empty(quat.shape[0],3).to(device)
                for ii in range(quat.shape[0]):
                    qt_target[ii] = torch.from_numpy(quaternion2eulerbytensor(quat[ii])).float()
                loss_qtx = mse_loss_grad(qt_target[:,0], qt_pred[:,0]) # huber_loss自己实现
                loss_qty = mse_loss_grad(qt_target[:,1], qt_pred[:,1])
                loss_qtz = mse_loss_grad(qt_target[:,2], qt_pred[:,2])
                loss = loss_qtx + loss_qty + loss_qtz
                loss_total += loss
                # logger.info('echo test epoch loss={:.4f}, loss_qtx={:.4}, loss_qty={:.4}, loss_qtz={:.4}'.format(loss, loss_qtx, loss_qty, loss_qtz))
            else:
                for ii in range(quat.shape[0]):
                    mseloss[ii] = nn.MSELoss()(qt_pred[ii], quat[ii])
                loss_total += torch.mean(mseloss)
                for ii in range(quat.shape[0]):
                    qt = quaternion2eulerbytensor(qt_pred[ii])                  
                    if quat[ii][0] == 0:
                        qt_target = numpy.zeros(3)
                    else:
                        qt_target = quaternion2eulerbytensor(quat[ii])
                    qtx = (qt[0] - qt_target[0])*180/math.pi
                    qty = (qt[1] - qt_target[1])*180/math.pi
                    qtz = (qt[2] - qt_target[2])*180/math.pi
                    avg_err = sum([abs(qtx), abs(qty), abs(qtz)])/3
                    # logger.info('output[{}] error(x,y,z)(degree) = {:.4f},{:.4f},{:.4f}, avg_err = {:.4f}, label = {}'.format(ii, qtx, qty, qtz, avg_err, label))
                    if avg_err < correct_threshold:
                        correct += 1
                    if avg_err < correct_threshold*2:
                        correct_2nd += 1
        # logger.info('epoch = {}, Accuracy on test set: {:.4f} = {}/{}, testLoss_epoch:{:.4f}'.format(epoch+1,100 * correct / total,  correct, total, loss_total/(index+1)))
        logger.info('epoch = {}, Accuracy on test set: {:.4f} = {}/{}, testLoss_epoch:{:.4f}, thetaerrmin:{:.6f},thetaerrmax:{:.6f},thetaerravg:{:.6f},phierrmin:{:.6f},phierrmax:{:.6f},phierravg:{:.6f}'.format(epoch+1,100*correct/total,  correct, total, loss_total/(index+1), thetaerr_min, thetaerr_max, thetaerr_avg/total, phierr_min, phierr_max, phierr_avg/total))
        
        tmp_accuracy = 100*correct/total
        if tmp_accuracy > best_accuracy:
            best_accuracy = tmp_accuracy
            state_dict = model.state_dict()
            torch.save(state_dict, './modelsave/bestmodel_5snr_acc_{:.1f}.pth'.format(best_accuracy))

        # with open('evaluate_error_case.txt', 'a') as f:
        #     f.write('============================' + '\n')



def gaussian_noise(tensor, mean=0, std=1):
    # 生成与输入tensor形状相同的高斯随机数
    noise = torch.randn_like(tensor) * std + mean
    # 将噪声添加到输入tensor上
    return tensor + noise

ADD_NOISE = True
# Evaluate the model on test data
def evaluatewithCNN(model,test_loader,device,epoch,num_layers, num_class, method, correct_threshold, noise_std,cnn_outpsize):
    model.eval()  # switch to evaluation mode

    with torch.no_grad():
        total = 0
        correct = 0
        loss_total = 0
        for index, (hrrp, label, quat, theta_phi, radlos) in enumerate(test_loader):
            hrrp = hrrp.to(device)
            quat = quat.to(device)
            label = label.to(device)
            theta_phi = theta_phi.to(device)

            hrrp = hrrp.float()
            if (ADD_NOISE):
                hrrp = gaussian_noise(hrrp, std=noise_std)

            cl_pred, qt_pred = model(hrrp)

            if method == 'multi_class_2out':
                total += theta_phi.shape[0]
            elif method == 'multi_class_3out':
                total += qt_pred.shape[0]
            else:
                total += quat.shape[0]

            
            if method == 'multi_class_3out':
                for ii in range(qt_pred.shape[0]):
                    # x = quat[int(numpy.floor(ii/cnn_outpsize))][0]
                    # y = quat[int(numpy.floor(ii/cnn_outpsize))][1]
                    # z = quat[int(numpy.floor(ii/cnn_outpsize))][2]
                    # w = quat[int(numpy.floor(ii/cnn_outpsize))][3]
                    # roll = (torch.atan(2*(z*y+w*x)/(w**2-x**2-y**2+z**2)) -  qt_pred[ii][0])*180/torch.pi #
                    # pith = (torch.asin(2*(w*y-x*z)) - qt_pred[ii][1])*180/torch.pi
                    # yaw = (torch.atan(2*(x*y+w*z)/(w**2+x**2-y**2-z**2)) - qt_pred[ii][2])*180/torch.pi
                    # avg_err = sum([abs(roll), abs(pith), abs(yaw)])/3
                    # logger.info('output[{}] error(x,y,z)(degree) = {:.4f},{:.4f},{:.4f}, avg_err = {:.4f}, label = {}'.format(ii, roll, pith, yaw, avg_err, label))
            
                    
                    qt_target = quaternion2euler(quat[int(numpy.floor(ii/cnn_outpsize))])
                    # qt_target = quaternion2euler(quat[ii])
                    qtx = (qt_pred[ii][0] - qt_target[0])*180/math.pi
                    qty = (qt_pred[ii][1] - qt_target[1])*180/math.pi
                    qtz = (qt_pred[ii][2] - qt_target[2])*180/math.pi
                    avg_err = sum([abs(qtx), abs(qty), abs(qtz)])/3
                    loss_total += avg_err
                    if avg_err < correct_threshold:
                        correct += 1
                    if avg_err < correct_threshold*2:
                        correct_2nd += 1
                loss_total = loss_total / ii
            else:
                for ii in range(quat.shape[0]):
                    qt = quaternion2euler(qt_pred[ii])                  
                    if quat[ii][0] == 0:
                        qt_target = numpy.zeros(3)
                    else:
                        qt_target = quaternion2euler(quat[ii])
                    qtx = (qt[0] - qt_target[0])*180/math.pi
                    qty = (qt[1] - qt_target[1])*180/math.pi
                    qtz = (qt[2] - qt_target[2])*180/math.pi
                    avg_err = sum([abs(qtx), abs(qty), abs(qtz)])/3
                    logger.info('output[{}] error(x,y,z)(degree) = {:.4f},{:.4f},{:.4f}, avg_err = {:.4f}, label = {}'.format(ii, qtx, qty, qtz, avg_err, label))
                    if avg_err < correct_threshold:
                        correct += 1
                    if avg_err < correct_threshold*2:
                        correct_2nd += 1

        logger.info('epoch = {}, Accuracy on test set: {:.4f} = {}/{}, testLoss_epoch:{:.4f}'.format(epoch+1,100 * correct / total,  correct, total, loss_total))




# Evaluate the model on test data
def evaluateClassification(model,test_loader,device,epoch,num_layers, num_class, method):
    model.eval()  # switch to evaluation mode

    with torch.no_grad():
        total = 0
        correct = 0
        for hrrp, label, quat, theta_phi, radlos in test_loader:
            hrrp = hrrp.to(device)
            quat = quat.to(device)
            label = label.to(device)
            theta_phi = theta_phi.to(device)

            hrrp = hrrp.float()
            cl_pred, qt_pred = model(hrrp)

            if method == 'multi_class_2out':
                total += theta_phi.shape[0]
            else:
                total += quat.shape[0]


            if num_class == 1:
                if method == 'multi_class_2out' or method == 'multi_class_2out_with_radarLos':
                    for ii in range(theta_phi.shape[0]):
                        theta = (qt_pred[ii][0] - theta_phi[ii][0])*180/math.pi
                        phi = (qt_pred[ii][1] - theta_phi[ii][1])*180/math.pi
                        avg_err = sum([abs(theta), abs(phi)])/2
                        logger.info('Traing output[{}] error(theta, phi)(degree) = {:.4f},{:.4f}, avg_err = {:.4f}'.format(ii, theta, phi, avg_err))
                        if avg_err < 1.5:
                            correct += 1
                elif method == 'multi_class_3out':
                    for ii in range(quat.shape[0]):
                        x = quat[ii][0]
                        y = quat[ii][1]
                        z = quat[ii][2]
                        w = quat[ii][3]
                        roll = (torch.atan(2*(z*y+w*x)/(w**2-x**2-y**2+z**2)) -  qt_pred[ii][0])*180/torch.pi #
                        pith = (torch.asin(2*(w*y-x*z)) - qt_pred[ii][1])*180/torch.pi
                        yaw = (torch.atan(2*(x*y+w*z)/(w**2+x**2-y**2-z**2)) - qt_pred[ii][2])*180/torch.pi
                        avg_err = sum([abs(roll), abs(pith), abs(yaw)])/3
                        # logger.info('output[{}] error(x,y,z)(degree) = {:.4f},{:.4f},{:.4f}, avg_err = {:.4f}, label = {}'.format(ii, roll, pith, yaw, avg_err, label))
                        if avg_err < 1.5:
                            correct += 1
                else:
                    for ii in range(quat.shape[0]):
                        qt = quaternion2euler(qt_pred[ii])                  
                        if quat[ii][0] == 0:
                            qt_target = numpy.zeros(3)
                        else:
                            qt_target = quaternion2euler(quat[ii])
                        qtx = (qt[0] - qt_target[0])*180/math.pi
                        qty = (qt[1] - qt_target[1])*180/math.pi
                        qtz = (qt[2] - qt_target[2])*180/math.pi
                        avg_err = sum([abs(qtx), abs(qty), abs(qtz)])/3
                        logger.info('output[{}] error(x,y,z)(degree) = {:.4f},{:.4f},{:.4f}, avg_err = {:.4f}, label = {}'.format(ii, qtx, qty, qtz, avg_err, label))
                        if avg_err < 1.5:
                            correct += 1

    logger.info('epoch = {}, Accuracy on test set: {:.4f} = {}/{} %'.format(epoch+1,100 * correct / total,  correct, total))

