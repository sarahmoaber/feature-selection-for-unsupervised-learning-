from numpy.random import rand
import numpy as np
import random
import math

#This function implements the PartGreedyFS score feature selection
# Reference: An Efficient Greedy Method for Unsupervised Feature Selection
# Input: number of partition = n_clusters, d = number of selected features

def PartGreedyFS(X, n_selected_features,n_clusters):
    n_samples, n_features = X.shape

    # initialize s
    S = list()

    # Generate a random parititioning of size p 
    P = random_partition(n_clusters, n_features)

    #Calculate B matrix of size n*len(P)
    B = create_B_matrix(X,P,n_samples)


    # Calculate B^T matrix of size len(P)*n 
    B_transpose = np.transpose(B)


    # initialize W= B^TX , size p*m 
    # to acess a specsific cloumn W[:,column index]
    W = np.dot(B_transpose, X)


    # initialize f , size m*1
    f = np.linalg.norm(W, axis=0)**2
    f = f.reshape(n_features, 1)


    # initialize g , size m*1

    g = np.zeros(n_features)
    for r in range(n_features):
       g[r] = np.dot( np.transpose(X[:,r]), X[:,r])
    g = g.reshape(n_features, 1)

    # initialize w , a list of length d, each tuple has m*1
    w = [[] for i in range(n_selected_features)]

    # initialize v , a list of length d, each tuple has p*1
    v = [[] for i in range(n_selected_features)]


    for t in range(n_selected_features):
        # l = arg max f_i / g_i
        # the rededundant features will have the same number !!
        out = np.divide(f, g)
        out = np.abs(out)
        max_value = np.nanmax(out)
        if (math.isnan (max_value)):
            return S
        index_of_maximum = np.where(out == max_value)
        l = index_of_maximum[0][0]
        

        # update s
        S.append(l)
        # print("The set now ," , S)


        #update segma of size m*1

        segma = np.dot(np.transpose(X),X[:,l])
        segma = segma.reshape(n_features, 1)

        sum = np.zeros(n_features)
        sum = sum.reshape(n_features, 1)
        for r in range(0,t-1):
            sum = sum + ((w[r])[l] * w[r])
        segma = np.subtract(segma, sum)

        # update gamma B^T*A_l  , size p*1
        gamma = W[:, l]
        gamma = gamma.reshape(len(P), 1)
        sum = np.zeros(len(P))
        sum = sum.reshape(len(P), 1)
        for r in range(0,t-1):
            sum = sum + ((w[r])[l] * v[r])
        gamma = np.subtract(gamma, sum)



        # update w_i , size m * 1 , access w using w[i] and w_l using (w[i])[l]
        w[t] = np.divide(segma, np.sqrt(np.abs(segma[l])))

        # update v_i , size c*1 , access v_i using v[i] and v_l^i using (v[i])[l]
        v[t] = np.divide(gamma, np.sqrt(np.abs(segma[l])))

        # update f , theorem 5
        sum = np.zeros(n_features)
        sum = sum.reshape(n_features, 1)
        for j in range(0,t-2):
            sum = sum + (np.dot(np.transpose(v[j]), v[t])) * w[j]
        f = f - (2 * (np.multiply(w[t], np.subtract(np.dot(np.transpose(W), v[t]), sum))) + (
            (np.linalg.norm(v[t])**2) * np.multiply(w[t], w[t])))

        # update g theorm 5
        g = g - np.multiply(w[t], w[t])
        f[S]= 0

    return S




# divide the features into p random parititions
def random_partition(p, n_features):
    P = [[] for i in range(p)]
    for x in range(n_features):
        P[random.randint(0, p-1)].append(x)
    return P


def create_B_matrix(X,P,n_samples):
    B = np.zeros((n_samples,len(P)))
    for j in P:
        idx = P.index(j)
        for r in j:
            B[:,idx] = B[:,idx] + X[:,r]
    return B         


