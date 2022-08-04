import numpy as np
import utilities

class make_SVD:
    def __init__(self, f_trace, trig_trace):
        # def where_trig(trig_trace):
        #     ## Boolean search for whether index n > n-1
        #     trig_onset = trig_trace[:-1] > trig_trace[1:]
        #     ## Get indices for where n > n-1
        #     self.trig_onset_index = np.where(trig_onset == 1)
        #     return self.trig_onset_index
        if trig_trace.ndim == 2:
            trig_trace = trig_trace[0]
        if len(f_trace) != len(trig_trace) and len(trig_trace) > len(f_trace):
            f_trace = utilities.data.interpolate(f_trace, len(trig_trace))
        def make_matrix(f_trace, trig_trace):
            ## Get locations of triggers
            trig_index = np.where(trig_trace == 1)[0]
            ## From this, derive the amount of triggers
            num_trigs = len(trig_index)
            ## Figure out how long the average inter-trigger duration is
            avg_trig_distance = round(np.average(np.gradient(trig_index)))
            ## Create an array with the correct dimensions
            segment_matrix = np.empty((num_trigs, avg_trig_distance))
            for n in range(num_trigs):
                ## Get current corresponding portion of f trace
                curr_segment = f_trace[trig_index[n]:trig_index[n] + avg_trig_distance]
                ## Baseline shift such that initial value is 0...
                ### Take the 1st value in each snippet
                first_val = curr_segment[0]
                ### Subtract that from the entire snippet
                curr_segment = curr_segment - first_val
                ## Insert segment into matrix accordingly
                try:
                    segment_matrix[n] = curr_segment
                except ValueError:
                    segment_matrix[n] = np.pad(curr_segment, (0, avg_trig_distance - len(curr_segment)))
            return segment_matrix
        ## Create attributes  
        self.matrix = make_matrix(f_trace, trig_trace)
        self.u, self.s, self.uv = np.linalg.svd(self.matrix)
        # if np.average(self.matrix) < 0:
        #     self.u = self.u * -1
        #     self.uv = self.uv * -1
        self.time, self.tuning = self.u[:, 0], self.uv[0] # Need to check that this is labled correctly
        # unit_u, unit_uv = self.u[:, 0], self.uv[0]
        ## Matrix multiplication for the first unit of u (u[:, 0]) and first unit of uv (uv[0])
        ### This gives us the reconstructed segment_matrix from only the prinicple unitary values 
        self.uxv = self.u[:, 0].reshape(len(self.u[:, 0]), 1) @ self.uv[0].reshape(1, len(self.uv[0]))
        ## Derive the normalized array
        self.uxv_norm = (self.uxv-np.min(self.uxv))/(np.max(self.uxv)-np.min(self.uxv))
        ## From the nomralized matrix, get maximum values of each array 
        # self.weightings = np.max(self.uxv_norm, axis = 1)
        # self.weightings = (self.u[0]-np.min(self.u[0]))/(np.max(self.u[0])-np.min(self.u[0]))
        self.weightings = self.u[:, 0]

        # # Old version
        # def make_matrix_interp(f_trace, trig_trace):
        #     interpolation_coefficient = 2 # Hard coded...
        #     ## Get locations of triggers
        #     self.trig_index = np.where(trig_trace == 1)[0]
        #     ## From this, derive the amount of triggers
        #     self.num_trigs = len(self.trig_index)
        #     ## Figure out how long the average inter-trigger duration is
        #     self.avg_trig_distance = round(np.average(np.gradient(self.trig_index)))
        #     ## Create an array with the correct dimensions
        #     matrix_to_be = np.empty((self.num_trigs-1, self.avg_trig_distance * interpolation_coefficient))
        #     ## Add an "estimated" trigger to the end of trig_frames (such that we also get the last response)
        #     value_to_add = self.trig_index[-1] + self.avg_trig_distance
        #     self.trig_index = np.append(self.trig_index, value_to_add)
        #     ## Get new number of trigs
        #     new_trig_index = len(self.trig_index)
        #     ## Secret params for testing whether interpolation is reliable 
        #     self.check_segs = []
        #     self.check_segs_intrp = []
        #     ## Loop through empty maytrix
        #     for n in range(self.num_trigs-1):
        #         ## Get the current corresponding portion of f trace
        #         curr_segment = f_trace[self.trig_index[n]:self.trig_index[n+1]]
        #         ## Append this to a secret param for testing whether interpolation is reliable 
        #         self.check_segs.append(curr_segment)
        #         ## Interpolate trace to ideal resolution (will occasionally be off bya frame or two)
        #         interp_curr_segemtn = utilities.data.interpolate(curr_segment, self.avg_trig_distance * interpolation_coefficient)
        #         ## Append for comparrison
        #         self.check_segs_intrp.append(interp_curr_segemtn)
        #         ## Set matrix indices accordingly
        #         matrix_to_be[n] = interp_curr_segemtn
        #     ## Interpolate whole matrix back to original temporal resolution
        #     # self.segmented_responses = utilities.data.interpolate(matrix_to_be, self.avg_trig_distance+1)
        #     self.segmented_responses = matrix_to_be
        #     return self.segmented_responses
