# For checking out stuff.

# # A generator in julia:
# function generator()
#     Channel() do channel
#         for i in 0:5
#             put!(channel, Char('A' + i))
#         end
#     end
# end

# g = generator()
# function Mtemp_gen(q)
#     ret = -1
#     try
#         ret = take!(g)
#     catch e
#         println("Sorry, no more elements in generator (channel)")
#         # @printf "Sorry, no more elements in generator (channel)\n"
#     end
# end

# Mtemp_gen(5)


# k = 0
# for i in 1:10
#     try
#         @printf "%c\n" take!(g)
#         @printf "At the %d'th entry\n" k
#         k += 1
#     catch e
#         println("No more!!")
#         k += 1
#     end
# end