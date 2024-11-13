from .ClassifierStrategy import ClassifierStrategy, ClassifierContext

class ClassifierFactory:
    _strategies = {}

    @classmethod
    def register_strategy(cls, classification_type, strategy):
        if not issubclass(strategy, ClassifierStrategy):
            raise TypeError('Strategy must be a subclass of ClassifierStrategy')
        cls._strategies[classification_type] = strategy

    @classmethod
    def get_classifier(cls, classification_type, classifier_input):
        if classification_type not in cls._strategies:
            raise ValueError('Invalid classification type.')
        args = ClassifierStrategy.parse_arguments()
        args.input = classifier_input
        strategy = cls._strategies[classification_type](args)
        return ClassifierContext(strategy).classifier