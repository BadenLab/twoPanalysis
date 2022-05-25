import numpy as np
import utilities

class make_SVD:
    def __init__(self, f_trace, trig_trace):
        def where_trig(trig_trace):
            ## Boolean search for whether index n > n-1
            trig_onset = trig_trace[:-1] > trig_trace[1:]
            ## Get indices for where n > n-1
            self.trig_onset_index = np.where(trig_onset == 1)
            return self.trig_onset_index
        # def 
        def make_matrix(f_trace, trig_trace):
            interpolation_coefficient = 2 # Hard coded...
            ## Get locations of triggers
            self.trig_index = where_trig(trig_trace)
            ## From this, derive the amount of triggers
            self.num_trigs = len(self.trig_index[0])
            ## Figure out how long the average inter-trigger duration is
            self.avg_trig_distance = round(np.average(np.gradient(self.trig_index[0])))
            ## Create an array with the correct dimensions
            matrix_to_be = np.empty((self.num_trigs-1, self.avg_trig_distance * interpolation_coefficient))
            ## Add an "estimated" trigger to the end of trig_frames (such that we also get the last response)
            value_to_add = self.trig_index[0][-1] + self.avg_trig_distance
            self.trig_index = np.append(self.trig_index, value_to_add)
            ## Get new number of trigs
            new_trig_index = len(self.trig_index)
            ## Secret params for testing whether interpolation is reliable 
            self.check_segs = []
            self.check_segs_intrp = []
            ## Loop through empty maytrix
            for n in range(self.num_trigs-1):
                ## Get the current corresponding portion of f trace
                curr_segment = f_trace[self.trig_index[n]:self.trig_index[n+1]]
                ## Append this to a secret param for testing whether interpolation is reliable 
                self.check_segs.append(curr_segment)
                ## Interpolate trace to ideal resolution (will occasionally be off bya frame or two)
                interp_curr_segemtn = utilities.data.interpolate(curr_segment, self.avg_trig_distance * interpolation_coefficient)
                ## Append for comparrison
                self.check_segs_intrp.append(interp_curr_segemtn)
                ## Set matrix indices accordingly
                matrix_to_be[n] = interp_curr_segemtn
            ## Interpolate whole matrix back to original temporal resolution
            # self.segmented_responses = utilities.data.interpolate(matrix_to_be, self.avg_trig_distance+1)
            self.segmented_responses = matrix_to_be
            return self.segmented_responses
        self.matrix = make_matrix(f_trace, trig_trace)
        self.u, self.s, self.uv = np.linalg.svd(self.matrix)
        unit_u, unit_uv = self.u[:, 0], self.uv[0]
        self.uxv = unit_uv.reshape(len(unit_uv), 1) @ unit_u.reshape(1, len(unit_u))