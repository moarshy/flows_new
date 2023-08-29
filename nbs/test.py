import ray
import logging

logger = logging.getLogger(__name__)
ray.init()

class Component:
    def execute(self):
        pass

    @ray.remote
    def pexecute(self):
        self.execute()

class Block:
    def __init__(self, components, type='serial'):
        self.components = components
        self.type = type

    def execute(self):
        if self.type == 'serial':
            for component in self.components:
                component.execute()
        elif self.type == 'parallel':
            futures = [component.pexecute.remote() for component in self.components]
            ray.get(futures)

class PrintComponent(Component):
    def __init__(self, message):
        self.message = message

    def execute(self):
        logger.info(self.message)

if __name__ == '__main__':
    component1 = PrintComponent("Hello, World!")
    component2 = PrintComponent("Goodbye, World!")

    logger.info("Executing components serially")
    block = Block([component1, component2], type='serial')
    block.execute()

    logger.info("Executing components in parallel")
    block = Block([component1, component2], type='parallel')
    block.execute()
