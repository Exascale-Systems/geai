from simpeg import potential_fields


class GravitySimulator:
    def __init__(self, mesh, ind_active, model_map, true_model, blocks_mask=None):
        self.mesh = mesh
        self.ind_active = ind_active
        self.model_map = model_map
        self.true_model = true_model
        self.blocks_mask = blocks_mask
        self.simulation = None
        self.survey = None
        self.user_data = {}  # For any extra metadata like receiver locations

    def run_survey(self, survey, engine="choclo"):
        self.survey = survey
        self.simulation = potential_fields.gravity.simulation.Simulation3DIntegral(
            survey=survey,
            mesh=self.mesh,
            rhoMap=self.model_map,
            active_cells=self.ind_active,
            engine=engine,
        )
        return self.simulation.dpred(self.true_model)
