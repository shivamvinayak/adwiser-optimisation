#!/usr/bin/env python
# coding: utf-8

from scipy.optimize import minimize
from numpy import log as ln
import numpy as np



def obj_fun(x,alpha,link,beta):
    
    #alpha = [0.23, 0.65]
            
    #the variables
    x1  = x[0]
    x2  = x[1]
    x3  = x[2]
    x4  = x[3]
    x12 = x[4]
    x13 = x[5]
    x14 = x[6]
    x23 = x[7]
    x24 = x[8]
    x34 = x[9]
    x123 = x[10]
    x124 = x[11]
    x134 = x[12]
    x234 = x[13]
    x1234 = x[14]
    
    #isolated throughput
    beta1 = beta[0]
    beta2 = beta[1]
    beta3 = beta[2]
    beta4 = beta[3] 

    #returns the objective function as per the elected link to the scipy solver   
    if link == "1100":#links 1 & 2 active
        return -(ln(x1*beta1 + x12*alpha[0]*beta1) + ln(x2*beta2 + x12*alpha[3]*beta2))
    
    elif link == "1010":#links 1 & 3 active
        return -(ln(x1*beta1 + x13*alpha[1]*beta1) + ln(x3*beta3 + x13*alpha[6]*beta3))
    
    elif link == "1001":#links 1 & 4 active
        return -(ln(x1*beta1 + x14*alpha[2]*beta1) + ln(x4*beta4 + x14*alpha[9]*beta4))
    
    elif link == "0110":#links 2 & 3 active
        return -(ln(x2*beta2 + x23*alpha[4]*beta2) + ln(x3*beta3 + x23*alpha[7]*beta3))
    
    elif link == "0101":#links 2 & 4 active
        return -(ln(x2*beta2 + x24*alpha[5]*beta2) + ln(x4*beta4 + x24*alpha[10]*beta4))
    
    elif link == "0011":#links 3 & 4 active
        return -(ln(x3*beta3 + x34*alpha[8]*beta3) + ln(x4*beta4 + x34*alpha[11]*beta4)) 
    
    elif link == "1110":#links 1, 2 & 3 active
        return -(ln(x1*beta1 + x12*alpha[0]*beta1 + x13*alpha[1]*beta1 + x123*alpha[12]*beta1)
               + ln(x2*beta2 + x12*alpha[3]*beta2 + x23*alpha[4]*beta2 + x123*alpha[15]*beta2)
               + ln(x3*beta3 + x13*alpha[6]*beta3 + x23*alpha[7]*beta3 + x123*alpha[18]*beta3))
    
    elif link == "1101":# links 1, 2 & 4 active
        return -(ln(x1*beta1 + x12*alpha[0]*beta1 + x14*alpha[2]*beta1 + x124*alpha[13]*beta1)
               + ln(x2*beta2 + x12*alpha[3]*beta2 + x24*alpha[5]*beta2 + x124*alpha[16]*beta2)
               + ln(x4*beta4 + x14*alpha[9]*beta4 + x24*alpha[10]*beta4 + x124*alpha[21]*beta4))
    
    elif link == "1011": #links 1, 3 & 4 active        
        return -(ln(x1*beta1 + x13*alpha[1]*beta1 + x14*alpha[2]*beta1 + x134*alpha[14]*beta1)
               + ln(x3*beta3 + x13*alpha[6]*beta3 + x34*alpha[8]*beta3 + x134*alpha[19]*beta3)
               + ln(x4*beta4 + x14*alpha[9]*beta4 + x34*alpha[11]*beta4 + x134*alpha[22]*beta4))
    
    elif link == "0111": #links 1, 3 & 4 active
        return -(ln(x2*beta2 + x23*alpha[4]*beta2 + x24*alpha[5]*beta2 + x234*alpha[17]*beta2)
               + ln(x3*beta3 + x23*alpha[7]*beta3 + x34*alpha[8]*beta3 + x234*alpha[20]*beta3)
               + ln(x4*beta4 + x24*alpha[10]*beta4 + x34*alpha[11]*beta4 + x234*alpha[23]*beta4))
        
    elif link == "1111": #links 1, 2, 3 & 4 active
        return -(ln(x1*beta1 + x12*alpha[0]*beta1 + x13*alpha[1]*beta1 + x14*alpha[2]*beta1 + 
                x123*alpha[12]*beta1 + x124*alpha[13]*beta1 + x134*alpha[14]*beta1 + x1234*alpha[24]*beta1)
             + ln(x2*beta2 + x12*alpha[3]*beta2 + x23*alpha[4]*beta2 + x24*alpha[5]*beta2 + 
                x123*alpha[15]*beta2 + x124*alpha[16]*beta2 + x234*alpha[17]*beta2 + x1234*alpha[25]*beta2)
             + ln(x3*beta3 + x13*alpha[6]*beta3 + x23*alpha[7]*beta3 + x34*alpha[8]*beta3 + 
                x123*alpha[18]*beta3 + x134*alpha[19]*beta3 + x234*alpha[20]*beta3 + x1234*alpha[26]*beta3)
             + ln(x4*beta4 + x14*alpha[9]*beta4 + x24*alpha[10]*beta4 + x34*alpha[11]*beta4 + 
                x124*alpha[21]*beta4 + x124*alpha[22]*beta4 + x234*alpha[23]*beta4 + x1234*alpha[27]*beta4))
    
def constraint(x):
    # sum x_i = 1
    return sum(x) - 1

if __name__ == "__main__":
    #initialize the variables with random values
    x0 = np.random.random_sample(size = 15)
    x0 /= x0.sum()
    #print(obj_fun(x0))

    #bounds for variables
    b = [0.0,1.0]
    bound = [b]*15
    
    beta = [101.71, 103.8, 78.06, 122.04]
    
    ##select the active links here
    ## 1100 -> links 1 & 2 active
    ## 1010 -> Links 1 & 3 active
    ## 1001 -> Links 1 & 4 active
    ##..........................
    ##..........................
    ## 0111 -> Links 2, 3 & 4 active
    ## 1111 -> Links 1, 2, 3 & 4 active
    
    link = "1111"
    
    m12 = [50,50,0,0]
    m13 = [101,0,39,0]
    m14 = [101,0,0,122]
    m23 = [0,30,30,0]
    m24 = [0,25,0,80]
    m34 = [0,0,29,80]
    
    m123 = [100,2,37,0]
    m124 = [80,20,0,50]
    m134 = [80,0,20,50]
    m234 = [0,20,30,50]
    
    m1234 = [50,2,10,40]
    
    
    # the parameter alphai_m the factor by which the throughput
    # of link i drops when links in set m are active.
    # m belongs to power set of {1,2,3,4} excluding the null sets. 
    alpha = np.array([])
    
    alpha = np.append(alpha,m12[0]/beta[0])
    alpha = np.append(alpha,m13[0]/beta[0])
    alpha = np.append(alpha,m14[0]/beta[0])
    alpha = np.append(alpha,m12[1]/beta[1])
    alpha = np.append(alpha,m13[1]/beta[1])
    alpha = np.append(alpha,m24[1]/beta[1])
    alpha = np.append(alpha,m13[2]/beta[2])
    alpha = np.append(alpha,m23[2]/beta[2])
    alpha = np.append(alpha,m34[2]/beta[2])
    alpha = np.append(alpha,m14[3]/beta[3])
    alpha = np.append(alpha,m24[3]/beta[3])
    alpha = np.append(alpha,m34[3]/beta[3])
    
    alpha = np.append(alpha,m123[0]/beta[0])
    alpha = np.append(alpha,m124[0]/beta[0])
    alpha = np.append(alpha,m134[0]/beta[0])
    alpha = np.append(alpha,m123[1]/beta[1])
    alpha = np.append(alpha,m124[1]/beta[1])
    alpha = np.append(alpha,m234[1]/beta[1])
    alpha = np.append(alpha,m123[2]/beta[2])
    alpha = np.append(alpha,m134[2]/beta[2])
    alpha = np.append(alpha,m234[2]/beta[2])
    alpha = np.append(alpha,m124[3]/beta[3])
    alpha = np.append(alpha,m134[3]/beta[3])
    alpha = np.append(alpha,m234[3]/beta[3])
    
    alpha = np.append(alpha,m1234[0]/beta[0])
    alpha = np.append(alpha,m1234[1]/beta[1])
    alpha = np.append(alpha,m1234[2]/beta[2])
    alpha = np.append(alpha,m1234[3]/beta[3])
    
    
    #add the constraint to solver
    cons = {'type':'eq','fun': constraint}

    pars = (alpha,link,beta,)

    #obtain the objective function & passed to scipy solver
    sol = minimize(obj_fun,x0, args=(pars),method='SLSQP',bounds=bound, constraints=cons)


    # print the vector x, containing optimal values of x_i
    print(sol)
