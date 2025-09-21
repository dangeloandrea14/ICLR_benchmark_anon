import copy

from erasure.core.factory_base import get_instance_config
from erasure.core.measure import Measure
from erasure.evaluations.evaluation import Evaluation
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local


class MembershipInference(Measure):
    """ Membership Inference Attack (MIA)
        abstract class that creates shadow models
    """

    '''def __init__(self, global_ctx: Global, local_ctx):
        super().__init__(global_ctx, local_ctx)'''
    def init(self):
        super().init()
        self.n_shadows = self.local.config['parameters']['shadows']['n_shadows']
        self.data_out_path = self.local.config['parameters']['data_out_path']+'_'+str(self.global_ctx.config.globals['seed'])
        self.train_part_plh = self.local.config['parameters']['shadows']['train_part_plh']
        self.test_part_plh = self.local.config['parameters']['shadows']['test_part_plh']
        #self.attack_test_part = self.local.config['parameters']['attack_test_part']
        self.base_model_cfg = self.params["shadows"]["base_model"]

        self.local_config["parameters"]["attack_in_data"]["parameters"]['DataSource']["parameters"]['path'] += '_'+str(self.global_ctx.config.globals['seed'])

        self.attack_in_data_cfg = self.local_config["parameters"]["attack_in_data"]

        self.forget_part = self.params["forget_part"]

        self.params["loss_fn"]["parameters"]["reduction"] = "none"
        self.loss_fn = get_instance_config(self.params['loss_fn'])


        data_cfg = self.local.config['parameters']['shadows']['shadow_in_data']
        data_cfg['parameters']['partitions'] += self.local.config['parameters']['shadows']['dataset_preproc']
        self.attack_in_data_cfg['generated_from']=data_cfg

        self.dataset = self.global_ctx.factory.get_object(Local(data_cfg))


        self.base_model_cfg['per_shadows_partition']=self.local.config['parameters']['shadows']['per_shadows_partition'] #Let the cache working  when per_partition is changed
        self.attack_in_data_cfg['generated_from_per_shadows']=self.base_model_cfg['per_shadows_partition'] #Let the cache working  when per_partition is changed


        # Shadow Models
        shadow_models = []
        for k in range(self.n_shadows):
            self.info(f"Creating shadow model {k}")
            self.dataset.add_partitions(copy.deepcopy([self.local.config['parameters']['shadows']['per_shadows_partition']]), "_"+str(k))
            shadow_models.append(self.create_shadow_model(k))

        # Attack DataManagers
        attack_datasets = self.create_attack_datasets(shadow_models)

        # Attack Models
        self.attack_models = {}
        for c, dataset in attack_datasets.items():
            self.info(f"Creating attack model {c}")
            current = Local(self.local_config['parameters']['attack_model'])
            current.dataset = dataset
            self.attack_models[c] = self.global_ctx.factory.get_object(current)

    def check_configuration(self):
        super().check_configuration()

        if "base_model" not in self.params["shadows"]:
            self.params["shadows"]["base_model"] = copy.deepcopy(self.global_ctx.config.predictor) #TODO: cache not fully work if copyed orgiinal

        if "shadow_in_data" not in self.params["shadows"]:
            self.local.config['parameters']['shadows']['shadow_in_data']=copy.deepcopy(self.global_ctx.config.data)

        if "loss_fn" not in self.params:
            self.params["loss_fn"] = copy.deepcopy(self.global_ctx.config.predictor["parameters"]["loss_fn"])

        self.params['forget_part'] = self.params.get('forget_part', 'forget')

    def process(self, e: Evaluation):
        return e

    def create_shadow_model(self, k):
        """ create generic Shadow Model """
        # create shadow model
        shadow_base_model = copy.deepcopy(self.base_model_cfg)
        shadow_base_model['current_shadow']=k
        shadow_base_model['parameters']['training_set'] = self.train_part_plh +"_"+str(k)
        current = Local(shadow_base_model)

        current.dataset = self.dataset
        shadow_model = self.global_ctx.factory.get_object(current)

        return shadow_model

    def create_attack_datasets(self, shadow_models):
        return {}