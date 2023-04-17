from neural import NeuralNet

print("\n\nTraining SQ\n\n")
sq_training_data = [
    ([0.2], [0.04]),
    ([0.3], [0.09]),
    ([0.5], [0.25]),
    ([0.7], [0.49]),
    ([0.1], [0.01]),
]
sqn = NeuralNet(1, 6, 1)
sqn.train(sq_training_data)


xor_data=[
    ([0,0],[0]),
    ([0,1],[1]),
    ([1,0],[1]),
    ([1,1],[0])
    ]
print()
print("xor data")
print()
orn=NeuralNet(2,1,1)
orn.train(xor_data)
print(orn.test_with_expected(xor_data))

print()
print("\n\nTraining voter opinion\n\n")
print()

voter_opinion= [
    ([.9,.6,.8,.3,.1],[1]),
    ([.8,.8,.4,.6,.4],[1]),
    ([.7,.2,.4,.6,.3],[1]),
    ([.5,.5,.8,.4,.8],[0]),
    ([.3,.1,.6,.8,.8],[0]),
    ([.6,.3,.4,.3,.6],[0])
]

von = NeuralNet(5,100,1)

von.train(voter_opinion)
print(von.test_with_expected(voter_opinion))

#test data
print()
print(f"case 1: {von.evaluate([1,1,1,.1,.1])}")
print(f"case 2: {von.evaluate([.5,.2,.1,.7,.7])}")
print(f"case 3: {von.evaluate([.8,.3,.3,.3,.8])}")
print(f"case 4: {von.evaluate([.8,.3,.3,.8,.3])}")
print(f"case 5: {von.evaluate([.9,.8,.8,.3,.6])}")

