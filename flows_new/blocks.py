import ray
from flows_new.config import get_logger

logger = get_logger(__name__)


class Block:
    def __init__(self, components, type='serial'):
        self.components = components
        self.type = type

    def execute(self):
        if self.type == 'serial':
            for i, component in enumerate(self.components):
                if i > 0 and self.components[i].expects_input:
                    component.input_from_prev = self.components[i-1].output
                component.execute()

        elif self.type == 'parallel':
            futures = [component.pexecute.remote(component) for component in self.components]
            outputs = ray.get(futures)
            for component, output in zip(self.components, outputs):
                component.output = output