import numpy as np
import pandas as pd
import numbers
# import sklearn.linear_model
# import tick.hawkes as th
import copy
# from base import kernels
import quadprog


class Hawkes:
    # functions of the class"
    # simulation
    # fast_fit
    # evaluation RMSE

    def __init__(self):
        self.generated_time = []
        self.sim_parameters = []
        self.input_list = []
        self.fit_parameters = []
        self.simulated = False
        self.fitted = False

    def multi_base_generate(self):
        def base_generate(mu):
            # generate one dimension exogenous event
            t = 0

            def Ft_inv(u):
                return -np.log(1 - u) / mu

            X = []
            while t <= self.sim_parameters['T']:
                rand = np.random.rand()
                t = t + Ft_inv(rand)
                X.append(t)
            del X[-1]
            return X

        output = []
        for i in np.arange(self.sim_parameters['dimension']):
            output.append(base_generate(float(self.sim_parameters['mu'][i])))
        return output

    def cluster_diffusion(self, exogenous_event):
        # exogenous_event should be generated through multi_base_generate function.
        # both alpha and beta are numpy matrices.

        alpha_2 = np.multiply(self.sim_parameters['alpha'], self.sim_parameters['beta'])
        beta_2 = self.sim_parameters['beta']

        def give_single_birth(parent_timestamp, alpha_ij, beta_ij):
            # give birth in one dimension
            # output is a cascade
            # we use thinning methods for this part
            output = []
            output_star = [0]
            # u = np.random.uniform(0, 1-np.exp(-alpha* np.exp(-beta*output[-1])))
            # make sure the arrival interval won't be infinity

            while True:
                # u = np.random.uniform(0, 1 - np.exp(-alpha_ij * np.exp(-beta_ij * output[-1])))
                u = np.random.rand()

                # if Ft_inv(u, output[-1], alpha_ij, beta_ij) == 0:
                # break
                if output_star[-1] - np.log(1 - u) + parent_timestamp < self.sim_parameters['T']:
                    output_star.append(output_star[-1] - np.log(1 - u) / alpha_ij)
                else:
                    break

            del output_star[0]

            for i in output_star:
                u = np.random.rand()
                if u <= 1 / beta_ij * np.exp(-beta_ij * i):
                    output.append(i)

            return np.asarray(output) + parent_timestamp

        def give_birth(parent_timestamp, k):
            direct_offspring_timestamp = []
            for i in np.arange(self.sim_parameters['dimension']):
                direct_offspring_timestamp.append([])
            for i in np.arange(self.sim_parameters['dimension']):
                a = give_single_birth(parent_timestamp,
                                      alpha_2[int(i), int(k)],
                                      beta_2[int(i), int(k)])
                direct_offspring_timestamp[i].append(a)
            return [direct_offspring_timestamp, parent_timestamp]

        output_list = []  # used to save list-like endogenous event cascades
        parent_list = []  # same shape with output_list, saving parent timestamp
        for i in np.arange(self.sim_parameters['dimension']):  # construct data structure
            output_list.append([])
            parent_list.append([])
        exogenous_event = np.array(exogenous_event)
        exo_timestamps = np.concatenate(exogenous_event)
        exo_dimensions = []
        for k in np.arange(self.sim_parameters['dimension']):
            exo_dimensions.append(np.repeat(int(k), exogenous_event[k].__len__()))
        exo_dimensions = np.concatenate(exo_dimensions)
        fused_exo_cascade = pd.DataFrame({"1dimension": exo_dimensions,
                                          "2timestamp": exo_timestamps,
                                          "3parent": np.repeat(0, exo_timestamps.__len__())})
        fused_exo_cascade = fused_exo_cascade.sort_values(by="2timestamp")
        # fused_exo_cascade is a dataframe with 2 rows:
        # dimension and exogenous event timestamps

        #######################
        # firstly give birth to 1st generation in each dimension
        for k in np.arange(fused_exo_cascade.shape[0]):
            children = give_birth(fused_exo_cascade.iloc[int(k), 1],
                                  fused_exo_cascade.iloc[int(k), 0])
            # here children is a list with 2 elements,
            # the first one is a list wiht each element is an np.array.
            # the second one is a timestamp which is the parent_timestamp.
            for i in np.arange(self.sim_parameters['dimension']):
                if children[0][i][0].__len__() > 0:
                    for j in np.arange(children[0][i].__len__()):
                        if children[0][i][0][j] <= self.sim_parameters['T']:
                            output_list[i].append(children[0][i][0][j])
                            parent_list[i].append(children[1])

        endo_timestamps = np.concatenate(output_list)
        endo_parents = np.concatenate(parent_list)
        endo_dimensions = []
        for k in np.arange(self.sim_parameters['dimension']):
            endo_dimensions.append(np.repeat(int(k), output_list[k].__len__()))
        endo_dimensions = np.concatenate(endo_dimensions)
        fused_endo_cascade = pd.DataFrame({"1dimension": endo_dimensions,
                                           "2timestamp": endo_timestamps,
                                           "3parent": endo_parents})
        fused_endo_cascade = fused_endo_cascade.sort_values(by="2timestamp")

        # secondly using endogenous data to generate other events.
        k = 0
        while (k < fused_endo_cascade.shape[0]):
            children = give_birth(fused_endo_cascade.iloc[k, 1],
                                  fused_endo_cascade.iloc[k, 0])
            for i in np.arange(self.sim_parameters['dimension']):
                if children[0][i][0].__len__() > 0:
                    for j in np.arange(children[0][i][0].__len__()):
                        if children[0][i][0][j] <= self.sim_parameters['T']:
                            fused_endo_cascade.loc[-1] = [i, children[0][i][0][j], children[1]]
                            fused_endo_cascade.index = fused_endo_cascade.index + 1
            fused_endo_cascade = fused_endo_cascade.sort_values(by="2timestamp")
            k += 1
        return [fused_exo_cascade, fused_endo_cascade]

    def simulate(self, T, dimension, alpha, beta, mu, verbose = False):
        self.sim_parameters = {'T': T,
                               'dimension': dimension,
                               'alpha' : alpha,
                               'beta' : beta,
                               'mu':mu
                               }
        if hasattr(beta, '__len__'):
            self.sim_parameters['beta'] = beta
        elif isinstance(beta, numbers.Number):
            self.sim_parameters['beta'] = beta*np.ones([self.sim_parameters['dimension'], self.sim_parameters['dimension']])
        self.generated_time = []

        multi_base_sim = self.multi_base_generate()
        result = self.cluster_diffusion(multi_base_sim)
        output = result[0].append(result[1])
        output = output.sort_values(by='2timestamp')
        # input_list = []
        # for i in np.arange(self.sim_parameters['dimension']):
        #     input_list.append(list(output[output['1dimension'] == i]['2timestamp']))
        output.index = np.arange(output.shape[0])
        output['1dimension'] = output['1dimension'].astype('int32')
        self.generated_time = output
        self.simulated = True
        if verbose:
            return output

    def gradient(self, alpha, mu, steps, input_list = None, T = None, beta = None):
        if input_list is None:
            self.input_list = self.generated_time.copy()
        else:
            self.input_list = input_list
        global dimensionality
        dimensionality = np.unique(self.input_list['1dimension']).__len__()

        time_list = []
        for i in np.arange(dimensionality):
            time_list.append(list(self.input_list[self.input_list['1dimension'] == i]['2timestamp']))

        if T is None:
            if self.simulated:
                Tf = self.sim_parameters['T']
            else:
                Tf = np.max(time_list)

        def KernelExp(t, beta=1):
            return beta * np.exp(-beta * t)

        def exp_lasting_time(time_list, T, beta=1, kernel='exp'):
            output = copy.deepcopy(time_list)

            if kernel == 'exp':
                for seq in np.arange(time_list.__len__()):
                    for timestamp in np.arange(time_list[seq].__len__()):
                        output[seq][timestamp] = KernelExp((T - time_list[seq][timestamp]), beta)
            return output

        def z_function(time_list, T, beta = 1, kernel='exp'):
            exp_lasting_time_matrix = exp_lasting_time(time_list, T)
            dimensionality = time_list.__len__()
            output = np.zeros((dimensionality + 1, dimensionality + 1))

            output[0, 0] = T
            for i in np.arange(dimensionality):
                for k in np.arange(exp_lasting_time_matrix[i].__len__()):
                    output[0, i + 1] += (1. - exp_lasting_time_matrix[i][k])
                    output[i + 1, 0] += (1. - exp_lasting_time_matrix[i][k])

            for i in np.arange(dimensionality):
                for j in np.arange(dimensionality):
                    for k in np.arange(exp_lasting_time_matrix[i].__len__()):
                        for k_prime in np.arange(exp_lasting_time_matrix[j].__len__()):
                            lower_bound = np.abs(time_list[i][k] - time_list[j][k_prime])
                            output[i + 1, j + 1] += beta * (
                                        np.exp(-beta * lower_bound) - exp_lasting_time_matrix[i][k] *
                                        exp_lasting_time_matrix[j][k_prime]) / 2.
            return output

        def y_function(time_list, beta = 1, kernel='exp'):
            dimensionality = time_list.__len__()
            output = np.zeros((dimensionality + 1, dimensionality))

            for i in np.arange(dimensionality):
                #  output[0, i] = np.sum(time_list[i])
                output[0, i] = time_list[i].__len__()
            for i in np.arange(dimensionality):
                for j in np.arange(dimensionality):
                    for k in np.arange(time_list[i].__len__()):
                        k_prime_length = sum([t_k > time_list[i][k] for t_k in time_list[j]])
                        for k_prime in np.arange(k_prime_length) + (time_list[j].__len__() - k_prime_length):
                            output[i + 1, j] += KernelExp(time_list[j][k_prime] - time_list[i][k], beta)
            return output

        z_mat = z_function(time_list = time_list, T = Tf, beta = beta, kernel= 'exp')

        # if speedup:
        #     sample_flag = np.random.uniform(size=self.input_list.__len__())
        #     sampled_generated_time = self.input_list[sample_flag < P]
        #     time_list = []
        #     for i in np.arange(dimensionality):
        #         time_list.append(list(sampled_generated_time[sampled_generated_time['1dimension'] == i]['2timestamp']))

        y_mat = y_function(time_list = time_list, beta = beta, kernel= 'exp')

        theta_mat = np.concatenate((mu.reshape((1, dimensionality)), alpha.T))
        output = []
        for i in np.arange(steps.__len__()):
            theta_mat_i = theta_mat + steps[i]
            output.append(np.linalg.norm(np.matmul(z_mat, theta_mat_i) - y_mat))

        return output


    def fast_fit(self, input_list = None, T = None, beta = None, kernel='exp',
                 NonNeg=False, verbose = False, speedup=False, P = 0.2):
        if input_list is None:
            self.input_list = self.generated_time.copy()
        else:
            self.input_list = input_list
        global dimensionality
        dimensionality = np.unique(self.input_list['1dimension']).__len__()

        if speedup:
            sample_flag = np.random.uniform(size=self.input_list.__len__())
            sampled_generated_time = self.input_list[sample_flag < P]
            time_list = []
            for i in np.arange(dimensionality):
                time_list.append(list(sampled_generated_time[sampled_generated_time['1dimension'] == i]['2timestamp']))
        else:
            time_list = []
            for i in np.arange(dimensionality):
                time_list.append(list(self.input_list[self.input_list['1dimension'] == i]['2timestamp']))

        if T is None:
            if self.simulated:
                Tf = self.sim_parameters['T']
            else:
                Tf = np.max(time_list)

        def KernelExp(t, beta=1):
            return beta * np.exp(-beta * t)

        def exp_lasting_time(time_list, T, beta=1, kernel='exp'):
            output = copy.deepcopy(time_list)

            if kernel == 'exp':
                for seq in np.arange(time_list.__len__()):
                    for timestamp in np.arange(time_list[seq].__len__()):
                        output[seq][timestamp] = KernelExp((T - time_list[seq][timestamp]), beta)
            return output

        def z_function(time_list, T, beta = 1, kernel='exp'):
            exp_lasting_time_matrix = exp_lasting_time(time_list, T)
            dimensionality = time_list.__len__()
            output = np.zeros((dimensionality + 1, dimensionality + 1))

            output[0, 0] = T
            for i in np.arange(dimensionality):
                for k in np.arange(exp_lasting_time_matrix[i].__len__()):
                    output[0, i + 1] += (1. - exp_lasting_time_matrix[i][k])
                    output[i + 1, 0] += (1. - exp_lasting_time_matrix[i][k])

            for i in np.arange(dimensionality):
                for j in np.arange(dimensionality):
                    for k in np.arange(exp_lasting_time_matrix[i].__len__()):
                        for k_prime in np.arange(exp_lasting_time_matrix[j].__len__()):
                            lower_bound = np.abs(time_list[i][k] - time_list[j][k_prime])
                            output[i + 1, j + 1] += beta * (
                                        np.exp(-beta * lower_bound) - exp_lasting_time_matrix[i][k] *
                                        exp_lasting_time_matrix[j][k_prime]) / 2.
            return output

        def y_function(time_list, beta = 1, kernel='exp'):
            dimensionality = time_list.__len__()
            output = np.zeros((dimensionality + 1, dimensionality))

            for i in np.arange(dimensionality):
                #  output[0, i] = np.sum(time_list[i])
                output[0, i] = time_list[i].__len__()
            for i in np.arange(dimensionality):
                for j in np.arange(dimensionality):
                    for k in np.arange(time_list[i].__len__()):
                        k_prime_length = sum([t_k > time_list[i][k] for t_k in time_list[j]])
                        for k_prime in np.arange(k_prime_length) + (time_list[j].__len__() - k_prime_length):
                            output[i + 1, j] += KernelExp(time_list[j][k_prime] - time_list[i][k], beta)
            return output

        z_mat = z_function(time_list = time_list, T = Tf, beta = 1, kernel= 'exp')

        # if speedup:
        #     sample_flag = np.random.uniform(size=self.input_list.__len__())
        #     sampled_generated_time = self.input_list[sample_flag < P]
        #     time_list = []
        #     for i in np.arange(dimensionality):
        #         time_list.append(list(sampled_generated_time[sampled_generated_time['1dimension'] == i]['2timestamp']))

        y_mat = y_function(time_list = time_list, beta = 1, kernel= 'exp')

        # if speedup:
        #     P_matrix = np.diag(np.insert(np.ones(dimensionality)/P, 0, 1))
        #     P2_matrix = P_matrix @ np.ones([dimensionality + 1, dimensionality + 1]) @ P_matrix
        #     P2_matrix = P2_matrix * (np.ones([dimensionality + 1, dimensionality + 1]) - (1-P) * np.diag(np.ones(dimensionality + 1)))
        #     P2_matrix[0, 0] = 1
        #     z_mat = P2_matrix * z_mat
        #     y_mat = P_matrix @ y_mat / P

        if NonNeg == False:
            try:
                out_mat = np.linalg.solve(z_mat, y_mat)
            except:
                out_mat = np.linalg.solve(z_mat + 0.0001 * np.identity(z_mat.shape[0]), y_mat)
            if speedup:
                out_mat /= P
            self.fit_parameters = {'T': Tf,
                                   'alpha': out_mat[1:, :].T,
                                   'beta': beta,
                                   'mu': out_mat[0, :]
                                   }
            self.fitted = True
            if verbose:
                return out_mat[0, :], out_mat[1:, :].T

        elif NonNeg == True:
            G = -np.identity(dimensionality + 1)
            h = np.zeros(dimensionality + 1)
            out_mat = np.zeros((dimensionality + 1, dimensionality))

            for i in np.arange(dimensionality):
                sol = quadprog.solve_qp(z_mat, y_mat[:, i], -G, -h)
                #   out_mat[:, i] = sol[0].reshape((1, dimensiodnality + 1))[0]
                out_mat[:, i] = sol[0]
            if speedup:
                out_mat /= P
            self.fit_parameters = {'T': Tf,
                                   'alpha': out_mat[1:, :].T,
                                   'beta': beta,
                                   'mu': out_mat[0, :]
                                   }
            self.fitted = True
            if verbose:
                return out_mat[0, :], out_mat[1:, :].T
        else:
            raise ValueError('\'NonNeg\' is a boolean parameter')


    def evaluation(self, method = 'RMSE'):
        alpha = np.linalg.norm(self.sim_parameters['alpha'] - self.fit_parameters['alpha']) / self.sim_parameters['dimension']
        mu = np.linalg.norm(self.sim_parameters['mu'] - self.fit_parameters['mu']) / self.sim_parameters['dimension'] ** 0.5
        return {'alpha':alpha, 'mu':mu}





if __name__ == '__main__':
    m = 2
    T = 200
    mu = 0.5 * np.random.uniform(size = m)
    alpha =  0.5 * np.random.uniform(size = [m, m])
    hp = Hawkes()
#    hp.generate_thinning()  # using thinning method to generate synthetic data


    hp.simulate(T, dimension=m, alpha=alpha, beta=1, mu=mu)  # using cluster representation to generate synthetic data

    print(hp.generated_time.__len__())

    print(alpha)
    hp.fast_fit(verbose=True, speedup=True, P = 0.5)

    hp.fast_fit(verbose=True)

    hp.evaluation()
