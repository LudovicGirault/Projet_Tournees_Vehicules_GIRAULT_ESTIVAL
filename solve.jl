using CPLEX
include("io.jl")

### instance = readInputFile("data/n_6-euclidean_true")


function static_solve(inst::Inst)

	n = inst.n
	t = inst.t
	th = inst.th
	T = inst.T
	d = inst.d
	C = inst.C

### Definition du modele

	m = Model(with_optimizer(CPLEX.Optimizer))
	set_optimizer_attribute(m,"CPX_PARAM_TILIM",150)
 # Variables

	@variable(m,x[1:n,1:n],Bin)
	@variable(m,u[2:n] >= 0)

 # Objectif 
	@objective(m, Min, sum(t[i,j]*x[i,j] for i in 1:n,j in 1:n if j!=i))

 # Contraintes

#	@constraint(m, [i in 1:n], x[i,i] ==0)

	@constraint(m, [i in 2:n], sum(x[j,i] for j in 1:n if j!=i) == 1)
	@constraint(m, [i in 2:n], sum(x[i,j] for j in 1:n if j!=i) == 1)
	@constraint(m, sum(x[1,j] for j in 2:n) == sum(x[j,1] for j in 2:n))

	@constraint(m, [i in 2:n], u[i] <= C - d[i])
	for i in 2:n
		for j in 2:n
			if i!=j
				@constraint(m, u[j] - u[i] >= d[i] - C*(1 - x[i,j]))
			end
		end
	end
	@constraint(m, [j in 2:n], u[j] <= C*(1 - x[1,j]) ) 

	start = time()

	optimize!(m)


	return JuMP.primal_status(m) == JuMP.MathOptInterface.FEASIBLE_POINT, time() - start, round(objective_value(m)),MOI.get(m,MOI.RelativeGap())

end

function dual_solve(inst::Inst)

	n = inst.n
	t = inst.t
	th = inst.th
	T = inst.T
	d = inst.d
	C = inst.C
	x_sol = zeros(n,n)

### Definition du modele

	m = Model(with_optimizer(CPLEX.Optimizer))
	set_optimizer_attribute(m,"CPX_PARAM_TILIM",150)
 # Variables

	@variable(m,x[1:n,1:n],Bin)
	@variable(m,u[2:n] >= 0)
	@variable(m,beta1[1:n,1:n] >= 0)
	@variable(m,alpha1 >= 0)
	@variable(m,beta2[1:n,1:n] >= 0)
	@variable(m,alpha2 >= 0)


 # Objectif 
	@objective(m, Min, sum(t[i,j]*x[i,j] + beta1[i,j] + 2*beta2[i,j] for i in 1:n,j in 1:n if j!=i) + alpha1*T + alpha2*T*T)

 # Contraintes


	@constraint(m, [i in 2:n], sum(x[j,i] for j in 1:n if j!=i) == 1)
	@constraint(m, [i in 2:n], sum(x[i,j] for j in 1:n if j!=i) == 1)
	@constraint(m, sum(x[1,j] for j in 2:n) == sum(x[j,1] for j in 2:n))

	@constraint(m, [i in 2:n], u[i] <= C - d[i])
	for i in 2:n
		for j in 2:n
			if i!=j
				@constraint(m, u[j] - u[i] >= d[i] - C*(1 - x[i,j]))
			end
		end
	end

	for i in 1:n
		for j in 1:n
			if i!=j
				@constraint(m, alpha1 + beta1[i,j] >= (th[i] + th[j])*x[i,j] )
				@constraint(m, alpha2 + beta2[i,j] >= th[i]*th[j]*x[i,j])
			end
		end
	end


	@constraint(m, [j in 2:n], u[j] <= C*(1 - x[1,j]) ) 

	start = time()

	optimize!(m)

	for i in 1:n
		for j in 1:n
			if i!=j
				x_sol[i,j] = round(JuMP.value(x[i,j]))
			end
		end
	end

	return JuMP.primal_status(m) == JuMP.MathOptInterface.FEASIBLE_POINT, time() - start, objective_value(m), MOI.get(m,MOI.RelativeGap()), x_sol

end


function robust_coupant_solve(inst::Inst)

	start = time()

	n = inst.n
	t = inst.t
	th = inst.th
	T = inst.T
	d = inst.d
	C = inst.C

	contraintes = []
	optimal = false

	z_max = 0
	xcb = zeros(n,n)
	zcb = 0
	gap = 0

while !optimal && time()-start<= 300
	### Def problème maitre

	m = Model(with_optimizer(CPLEX.Optimizer))
	set_optimizer_attribute(m,"CPX_PARAM_TILIM",120)
	@variable(m,x[1:n,1:n],Bin)
	@variable(m,u[2:n] >= 0)
	@variable(m,z)

	@objective(m, Min, z)
	@constraint(m, z >= sum(t[i,j]*x[i,j] for i in 1:n,j in 1:n if j!=i))
	for e in contraintes
		delta1 = e[1]
		delta2 = e[2]
		@constraint(m, z>= sum(x[i,j]* (t[i,j] + delta1[i,j]*(th[i] + th[j]) + delta2[i,j]*th[i]*th[j]) for i in 1:n,j in 1:n if j!=i))
	end
	@constraint(m, [i in 2:n], sum(x[j,i] for j in 1:n if j!=i) == 1)
	@constraint(m, [i in 2:n], sum(x[i,j] for j in 1:n if j!=i) == 1)
	@constraint(m, sum(x[1,j] for j in 2:n) == sum(x[j,1] for j in 2:n))
	@constraint(m, [i in 2:n], u[i] <= C - d[i])
	for i in 2:n
		for j in 2:n
			if i!=j
				@constraint(m, u[j] - u[i] >= d[i] - C*(1 - x[i,j]))
			end
		end
	end
	@constraint(m, [j in 2:n], u[j] <= C*(1 - x[1,j]) ) 

	optimize!(m)

	xcb = zeros(n,n)
	zcb = Int64(round(objective_value(m)))
	gap = MOI.get(m,MOI.RelativeGap())
	for i in 1:n
		for j in 1:n
			if i!=j
				xcb[i,j] = round(JuMP.value(x[i,j]))
			end
		end
	end

### Def sous probleme
		s = Model(with_optimizer(CPLEX.Optimizer))
		set_optimizer_attribute(s,"CPX_PARAM_TILIM",120)
		@variable(s,1>=delta1[1:n,1:n]>=0)
		@variable(s,2>=delta2[1:n,1:n]>=0)
		@objective(s,Max,sum(xcb[i,j] * (t[i,j] + delta1[i,j]*(th[i] + th[j]) + delta2[i,j]*th[i]*th[j]) for i in 1:n,j in 1:n if j!=i))
		@constraint(s,sum(delta1[i,j] for i in 1:n,j in 1:n if j!=i) <= T)
		@constraint(s,sum(delta2[i,j] for i in 1:n,j in 1:n if j!=i) <= T*T)
		optimize!(s)
		z_max = Int64(round(objective_value(s)))
		if z_max > zcb
			println("-"^15," Ajout d'une coupe")
			delta1cb = zeros(n,n)
			delta2cb = zeros(n,n)
			for i in 1:n
				for j in 1:n
					delta1cb[i,j] = JuMP.value(delta1[i,j])
					delta2cb[i,j] = JuMP.value(delta2[i,j])
				end
			end
			contraintes = vcat(contraintes,[[delta1cb,delta2cb]])
		else 
			optimal = true
		end
		println("-"^15," Objectif problème maître courant : ", zcb)
		println("-"^15," Objectif sous-problème courant : ",objective_value(s))
end


	return true, time() - start, zcb, gap,xcb

end


function robust_lazy_solve(inst::Inst)
	start = time()
	n = inst.n
	t = inst.t
	th = inst.th
	T = inst.T
	d = inst.d
	C = inst.C

### Definition du problème maitre

	m = Model(with_optimizer(CPLEX.Optimizer))
		
	#OPTIONS CPLEX
	#-------------
	# Désactive le presolve (simplification automatique du modèle)
	set_optimizer_attribute(m, "CPXPARAM_Preprocessing_Presolve", 0)
	# Désactive la génération de coupes automatiques
	set_optimizer_attribute(m, "CPXPARAM_MIP_Limits_CutsFactor", 0)
	# Désactive la génération de solutions entières à partir de solutions
	# fractionnaires
	set_optimizer_attribute(m, "CPXPARAM_MIP_Strategy_FPHeur", -1)
	# Désactive les sorties de CPLEX (optionnel)
	set_optimizer_attribute(m, "CPX_PARAM_SCRIND", 0)

 	# Variables
	@variable(m,x[1:n,1:n],Bin)
	@variable(m,u[2:n] >= 0)
	@variable(m,z)
 	# Objectif 
	@objective(m, Min, z)
	# Contraintes
	@constraint(m, z >= sum(t[i,j]*x[i,j] for i in 1:n,j in 1:n if j!=i))
	@constraint(m, [i in 2:n], sum(x[j,i] for j in 1:n if j!=i) == 1)
	@constraint(m, [i in 2:n], sum(x[i,j] for j in 1:n if j!=i) == 1)
	@constraint(m, sum(x[1,j] for j in 2:n) == sum(x[j,1] for j in 2:n))
	@constraint(m, [i in 2:n], u[i] <= C - d[i])
	for i in 2:n
		for j in 2:n
			if i!=j
				@constraint(m, u[j] - u[i] >= d[i] - C*(1 - x[i,j]))
			end
		end
	end
	@constraint(m, [j in 2:n], u[j] <= C*(1 - x[1,j]) ) 

	
#### Fonction de callback

function mycallback(cb_data)

	xcb = zeros(n,n)
	zcb = callback_value.(Ref(cb_data), z)
	for i in 1:n
		for j in 1:n
			if i!=j
				xcb[i,j] = callback_value.(Ref(cb_data), x[i,j])
			end
		end
	end
	
### Def sous probleme
	s = Model(with_optimizer(CPLEX.Optimizer))
	@variable(s,1>=delta1[1:n,1:n]>=0)
	@variable(s,2>=delta2[1:n,1:n]>=0)
	@objective(s,Max,sum(xcb[i,j] * (t[i,j] + delta1[i,j]*(th[i] + th[j]) + delta2[i,j]*th[i]*th[j]) for i in 1:n,j in 1:n if j!=i))
	@constraint(s,sum(delta1[i,j] for i in 1:n,j in 1:n if j!=i) <= T)
	@constraint(s,sum(delta2[i,j] for i in 1:n,j in 1:n if j!=i) <= T*T)
	optimize!(s)
	
	if round(objective_value(s)) > zcb
		println("-"^15," Ajout d'une coupe")
		delta1cb = zeros(n,n)
		delta2cb = zeros(n,n)
		for i in 1:n
			for j in 1:n
				delta1cb[i,j] = JuMP.value(delta1[i,j])
				delta2cb[i,j] = JuMP.value(delta2[i,j])
			end
		end
		contrainte = @build_constraint(z >= sum(x[i,j] * (t[i,j] + delta1cb[i,j]*(th[i] + th[j]) + delta2cb[i,j]*th[i]*th[j]) for i in 1:n,j in 1:n if j!=i))
		MOI.submit(m,MOI.LazyConstraint(cb_data),contrainte)
	end
	println("-"^15," Objectif problème maître courant : ", zcb)
	println("-"^15," Objectif sous-problème courant : ",objective_value(s))
end
	MOI.set(m,MOI.LazyConstraintCallback(),mycallback)

	optimize!(m)

	return JuMP.primal_status(m) == JuMP.MathOptInterface.FEASIBLE_POINT, time() - start, round(objective_value(m)),MOI.get(m,MOI.RelativeGap())

end

function solve_all()
	dataFolder = "data/"
    	resFolder = "sol/"
#	resolutionMethod = "dual"   
#	resolutionMethod = "static"
#	resolutionMethod = "lazycut"
	resolutionMethod = "coupant"
#	resolutionMethod = "heuristic"
	resolutionFolder = resFolder .* resolutionMethod


    	global isOptimal = false
    	global solveTime = -1

	for file in readdir(dataFolder)
		println("-- Resolution of ", file)
		inst = readInputFile(dataFolder * file)
		
		if resolutionMethod == "dual"
			outputFile = resolutionFolder * "/" * file


                	fout = open(outputFile, "w")  

                	resolutionTime = -1
                	isOptimal = false

			isOptimal, resolutionTime,value,gap,x_sol = dual_solve(inst)
			if isOptimal
				println(fout,"isOptimal = ", isOptimal)
				println(fout,"cout = ",value)
				println(fout,"gap = ",gap)
				tour = tournees(x_sol)
				writetournees(fout,tour)
				println(fout,"solveTime = ",resolutionTime)
			else 
				println(fout,"isOptimal = ", isOptimal)
				println(fout,"solveTime = ",resolutionTime)
			end
			close(fout)
		elseif 	resolutionMethod == "coupant"
			outputFile = resolutionFolder * "/" * file


                	fout = open(outputFile, "w")  

                	resolutionTime = -1
                	isOptimal = false

			isOptimal, resolutionTime,value,gap, x = robust_coupant_solve(inst)
			if isOptimal
				println(fout,"isOptimal = ", isOptimal)
				println(fout,"cout = ",value)
				println(fout,"gap = ",gap)
				tour = tournees(x)
				writetournees(fout,tour)

				println(fout,"solveTime = ",resolutionTime)
			else 
				println(fout,"isOptimal = ", isOptimal)
				println(fout,"solveTime = ",resolutionTime)
			end
			close(fout)
		elseif  resolutionMethod =="heuristic"
			outputFile = resolutionFolder * "/" * file
                	fout = open(outputFile, "w") 
                	resolutionTime = -1
                	isOptimal = true

			value,tour,resolutionTime = heuristic_solve(inst)
			println(fout,"isOptimal = ", isOptimal)
			println(fout,"cout = ",value)
			writetournees(fout,tour)
			println(fout,"solveTime = ",resolutionTime)

		end

	end
end


function heuristic_solve(inst::Inst)
	start = time()
	n = inst.n
	t = inst.t
	th = inst.th
	T = inst.T
	d = inst.d
	C = inst.C

	deja_trie = zeros(Int64,n)
	deja_trie[1] = 1
	vaccins = 0
	copy_d = deepcopy(d)
	tournees = []
	i = 0
	while 0 in deja_trie
		vaccins = 0
		m = findfirst(isequal(maximum(copy_d)),copy_d)
		tournees=vcat(tournees,[[m]])
		i += 1
		vaccins += copy_d[m]
		deja_trie[m] = 1
		copy_d[m] = -1
		while vaccins < C
			max_ind = 1
			vaccins_max = vaccins + copy_d[max_ind]
			for j in 2:n
				if vaccins_max < vaccins + copy_d[j] && vaccins + copy_d[j] < C
					max_ind = j
					vaccins_max = vaccins + copy_d[j] 
				end
			end
			if max_ind == 1 
				vaccins = C
			else
				vaccins = vaccins_max
				tournees[i] = vcat(tournees[i],[max_ind])
				deja_trie[max_ind] = 1
				copy_d[max_ind] = -1
			end
		end	
	end
	cout = 0
	for tour in tournees
		taille = length(tour)
		cout += t[1,tour[1]]
		for i in 1:taille-1
			cout += t[tour[i],tour[i+1]]
		end
		cout += t[tour[taille],1]
	end
#	return cout,tournees,time() - start
	### Phase de descente 
	i = 0
	taille_tournee = length(tournees)
	while i < 1000
		current_tournees = deepcopy(tournees)
		new_cout = 0
		ind = rand(1:taille_tournee)
		tour = tournees[ind]
		taille = length(tour)
		client1 = rand(1:taille)
		client2 = rand(1:taille)
		current_tournees[ind][client1] = tournees[ind][client2]
		current_tournees[ind][client2] = tournees[ind][client1]
		for tour in current_tournees
			taille = length(tour)
			new_cout += t[1,tour[1]]
			for i in 1:taille-1
				new_cout += t[tour[i],tour[i+1]]
			end
			new_cout += t[tour[taille],1]
		end
		if new_cout < cout 
			tournees[ind][client1] = current_tournees[ind][client1]
			tournees[ind][client2] = current_tournees[ind][client2]
			cout = new_cout
		end
		i+=1
	end	

	return cout,tournees,time() - start
end
