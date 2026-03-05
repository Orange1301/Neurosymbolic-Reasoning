from z3 import *

def check_conclusion(premises, conclusion, timeout_ms=5000):
    """
    Checks if a conclusion is True, False, or Uncertain given premises.
    premises: A list of Z3 boolean expressions
    conclusion: A single Z3 boolean expression
    """
    s = Solver()
    s.set("timeout", timeout_ms)

    # s.add(premises & (-conclusion))
    for p in premises:
        s.add(p)
    s.push()
    s.add(Not(conclusion))
    
    result = s.check()
    
    if result == unsat:
        return "True"
    elif result == sat:
        return "False"
    else:
        return "Uncertain"

# --- EXAMPLE USAGE ---
# Define the domain
Object = DeclareSort('Object')
Human = Function('Human', Object, BoolSort())
Mortal = Function('Mortal', Object, BoolSort())
socrates = Const('socrates', Object)

x = Const('x', Object) # Logical variable for quantifiers

# Premises: All humans are mortal AND Socrates is a human
p1 = ForAll(x, Implies(Human(x), Mortal(x)))
p2 = Human(socrates)

# Conclusion: Socrates is mortal
concl = Mortal(socrates)

status = check_conclusion([p1, p2], concl)
print(f"The conclusion is: {status}")