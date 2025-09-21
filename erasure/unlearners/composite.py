from erasure.core.unlearner import Unlearner
from erasure.utils.config.local_ctx import Local

class Cascade(Unlearner):
    def init(self):
       
        super().init()

        sub_unlearner_cfg = self.local.config['parameters']['sub_unlearner']

        self.sub_unlearners = []

        for sub_un in sub_unlearner_cfg:
            current = Local(sub_un)
            current.dataset = self.dataset
            current.predictor = self.predictor
            self.sub_unlearners.append(self.global_ctx.factory.get_object(current))

    
    def unlearn(self):
        for unlrn in self.sub_unlearners:
            self.info('Execute Sub-Unlearner: '+unlrn.__class__.__name__ +' on '+ str(unlrn.predictor))
            unlrn.unlearn()

        return self.predictor
    
    def __unlearn__(self):
        pass

class Identity(Unlearner):
    
    def __unlearn__(self):
        return  self.predictor