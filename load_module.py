import importlib
import sys

class LoadModule:
   
    def __init__(self, module_folder : str ='/path/to/my/models/'):
        """
        Load module folder and whatever modules in it 
        """
        
        self.module_folder = module_folder            
    
        # import module folder
        sys.path.insert(0, self.module_folder)

    def import_module(self, module : str): 
        """
        Import modulefrom module foldder 
        """
        module = __import__(module)   
        
        
        return module
    
    def import_class(self, model : str, module : str = None, **kwargs):
        """
        Import a class from a module
        **kwargs : kwargs to load class directly if wished  
        """
        
        # if module not given module name is same as class
        if module is None:
            module = model
        
        module = self.import_module(module)
        
        model = getattr(module, model)
        if kwargs:
            return model(**kwargs)
        
        else:    
            return model
        
if __name__ == "__main__":    # how to use this 
    model_name = 'my_model' # Name of the script containing the class as well as the class (if different names then specify "module" also)
    models_path = '/path/to/my/models/' # path to the folder containing the model modules(s)
    load_models = LoadModule.LoadModule(models_path)
    
    kwargs_model = {'arg1' : True } # kwargs for modelt
    model = load_models.import_class(model_name, **kwargs_model)
