from abc import ABC

class Compartment:

    def __init__(self, **dimensions):
    
        self.length     = length
        self.width      = width
        self.height     = height
        self.max_weight = max_weight
        self.items      = items or []
    
standard = Compartment(**dict(
                            length=9*12,
                            width=42,
                            height=60,
                            max_weight=5060)
                      )

