"""
Handles the parsing of the config file 
and builds the necessary file structure for a given run set.


@author: Yash Gandhi

"""
import os
import yaml
import itertools
import numpy as np

from copy import deepcopy



# v. messy at the moment
class Config_Handler:
    def __init__(self, file):

        # load configurations from file
        self.config_file = file
        with open(self.config_file, 'rb') as f:
            self.config = yaml.load(f)

        # create folder to save model runs
        self.save_folder = os.path.join("saved_models/", self.config['run_name'])
        os.makedirs(self.save_folder, exist_ok=True)

        self.log_dir = os.path.join(self.save_folder, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)


        self.keys = self.config.keys()

    def _check_key(self, k):
        # valid key?
        if k not in self.keys:
            raise ValueError(str(k) +  " not a configuration parameter")
        else:
            return True

    def get_keys(self):
        """
         Get all keys 
        """
        return self._get_keys(self.config)        
    
    def _get_keys(self, d, prefix=None):
        keys = []
        for k in d.keys():
            keys.append([k])
            if(type(d[k]) is dict):
                for sub_k in self._get_keys(d[k], (k, )):
                    if(len(sub_k) > 1):
                        keys.append([k] + sub_k)
                    else:
                        keys.append([k] + sub_k)
        return keys

    def _get_obj(self, c, k):
        """
         Get an object based on a key (or set of keys for sub dictionaries)

         PARAMETERS:
         c (dictionary): dictionary to get the object from
         k (list<string>; string): key, or keys, to get the object 
        """
        if(type(k) is list and len(k) > 1):
            tmp = c
            for i in range(len(k) - 1):
                tmp = tmp[k[i]]
            return tmp[k[i + 1]]
        
        else:
            if(type(k) is list):
                return c[k[0]]
            else:
                return c[k]

    def _process_list(self, li):
        """
         Process a list in the configuration file
        """
        if(li[-1] == 'step'):
            if(len(li) != 4):
                raise ValueError(str(li) + 
                                " has been set as a step list and must follow [start, stop, step_value, \'step\']." + 
                                "\n\t\t Ex. [1,10,1,\'step\'] returns [1,2,3,4,5,6,7,8,9,10]. ")

            return np.arange(li[0], li[1] + li[2], li[2])
        
        elif(li[-1] == 'exp'):
            if(len(li) != 4):
                raise ValueError(str(li) + 
                                " has been set as a exp list and must follow [start, stop, rate, \'step\']." + 
                                "\n\t\t Ex. [1,10,.1,\'step\'] returns [.1^1, .1^2, ..., .1^10]. ")
            return li[2] ** np.arange(li[0], li[1], 1)
        elif(li[-1] == 'elements'):
            return np.array(li[:-1])

        else:
            raise ValueError(str(li[-1]) + ' needs to be either \'step\', \'exp\', \'elements\'. ')


    def _sub_obj(self, c, k, val):
        """
         sub the value at the k with given value
        """
        # Neat referencing trick so I don't have to reconstruct the 
        # dictionary in reverse! 
        holds = [c]
        tmp = c
        if(type(k) is list and len(k) > 1):
            for i in range(len(k) - 1):
                tmp = tmp[k[i]]
                holds.append(tmp)

            tmp[k[-1]] = val
            return holds[0]
        
        else:
            if(type(k) is list):
                c[k[0]] = val
            else:
                c[k] = val
        return c


    def generate_dictionaries(self):
        list_in_dict = []
        for k in self.get_keys():
            o = self._get_obj(self.config, k)
            if(type(o) is list):
                list_in_dict.append((k, self._process_list(o)))
        
        permutations = []
        for i in range(len(list_in_dict)):
            permutations.append(list_in_dict[i][1])

        sets_of_dicts = []
        perms = []
        for i, t in enumerate(itertools.product(*permutations)):
            tmp = deepcopy(self.config)
            perms.append(t)
            for kt in range(len(list_in_dict)):
                tmp = self._sub_obj(deepcopy(tmp), list_in_dict[kt][0], t[kt])
            sets_of_dicts.append(tmp)
        
        return sets_of_dicts



def test_config_handler(config):

    handler = Config_Handler(config)
    dicts = handler.generate_dictionaries()





if __name__ == "__main__":
    test_config_handler("core/test_config.yaml")
