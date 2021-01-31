using JuMP
using Plots
import GR

mutable struct Inst
	n::Int64
	t::Array{Int64,2}
	th::Array{Int64,1}
	T::Int64
	d::Array{Int64,1}
	C::Int64
end

function readInputFile(inputFile::String)


  # Open the input file
    datafile = open(inputFile)

    data = readlines(datafile)
    close(datafile)
    
    n = parse(Int64, split(data[1],"=")[2])

    #### lecture de t 

    t = Array{Int64,2}(undef,n,n)

    line = split(data[2],"[")[2]
    line = split(line, "]")[1]
    line = split(line, ";")
    for i in 1:n
	linesplit = split(line[i]," ")
#	println(linesplit)
        ind = 1
#	println(linesplit)
        for j in linesplit
		if !isempty(j)
	    		t[i,ind] = parse(Int64,j)
			ind +=1
		end
	end
    end

#    println("Alice ",t)

	### Lecture de th
 
    th = Array{Int64,1}(undef,n)

     line = split(data[3],"[")[2]
     line = split(line,"]")[1]
     line = split(line, ",")
     for i in 1:n
        th[i] = parse(Int64,line[i])
     end
#     println(th)

    T = parse(Int64, split(data[4],"=")[2])

	#### Lecture de d

     d = Array{Int64,1}(undef,n)
     line = split(data[5],"[")[2]
     line = split(line,"]")[1]
     line = split(line, ",")
     for i in 1:n
        d[i] = parse(Int64,line[i])
     end
#     println(d)

    C = parse(Int64, split(data[6],"=")[2])

	inst = Inst(n,t,th,T,d,C)

	return inst

end

function tournees(x::Array{Float64,2})

	tournees = []
	n = size(x,1)
	i = 0
	L1 = x[1,:]
	for j in 1:n
		if L1[j] == 1   ### nouvelle tournée detectée
			i +=1
			k = j
#			println(k)
			tournees = vcat(tournees,[[k]])
			while !(x[k,1] == 1)
#				println(x[k,:])
				k = Int64(findfirst(isequal(1.0),x[k,:]))
				tournees[i] = vcat(tournees[i],[k])
			end
		end
	end
	return tournees
end

function tournees(x::Array{Int64,2})

	tournees = []
	n = size(x,1)
	i = 0
	L1 = x[1,:]
	for j in 1:n
		if L1[j] == 1   ### nouvelle tournée detectée
			i +=1
			k = j
#			println(k)
			tournees = vcat(tournees,[[k]])
			while !(x[k,1] == 1)
#				println(x[k,:])
				k = Int64(findfirst(isequal(1.0),x[k,:]))
				tournees[i] = vcat(tournees[i],[k])
			end
		end
	end
	return tournees
end

function write(file::String,tournees::Array{Any,1})
	fout = open(file,"w")
	writetournees(fout,tournees)
	close(fout)


end

function writetournees(fout::IOStream,tournees::Array{Any,1})

	println(fout,"tournees = [")
	for t in tournees
		print(fout,"[1")
		for elt in t
			print(fout,",",elt)
		end
		println(fout,",1]")
	end
	println(fout,"];")
end


function performanceDiagram(outputFile::String)

    resultFolder = "tmp/"
    
    # Maximal number of files in a subfolder
    maxSize = 0

    # Number of subfolders
    subfolderCount = 0

    folderName = Array{String, 1}()

    # For each file in the result folder
    for file in readdir(resultFolder)

        path = resultFolder * file
        
        # If it is a subfolder
        if isdir(path)
            
            folderName = vcat(folderName, file)
             
            subfolderCount += 1
            folderSize = size(readdir(path), 1)

            if maxSize < folderSize
                maxSize = folderSize
            end
        end
    end

    # Array that will contain the resolution times (one line for each subfolder)
    results = Array{Float64}(undef, subfolderCount, maxSize)

    for i in 1:subfolderCount
        for j in 1:maxSize
            results[i, j] = Inf
        end
    end

    folderCount = 0
    maxSolveTime = 0

    # For each subfolder
    for file in readdir(resultFolder)
            
        path = resultFolder * file
        
        if isdir(path)

            folderCount += 1
            fileCount = 0

            # For each text file in the subfolder
            for resultFile in readdir(path)

                fileCount += 1
                include(path * "/" * resultFile)

                if isOptimal
                    results[folderCount, fileCount] = solveTime

                    if solveTime > maxSolveTime
                        maxSolveTime = solveTime
                    end 
                end 
            end 
        end
    end 

    # Sort each row increasingly
    results = sort(results, dims=2)

    println("Max solve time: ", maxSolveTime)

    # For each line to plot
    for dim in 1: size(results, 1)

        x = Array{Float64, 1}()
        y = Array{Float64, 1}()

        # x coordinate of the previous inflexion point
        previousX = 0
        previousY = 0

        append!(x, previousX)
        append!(y, previousY)
            
        # Current position in the line
        currentId = 1

        # While the end of the line is not reached 
        while currentId != size(results, 2) && results[dim, currentId] != Inf

            # Number of elements which have the value previousX
            identicalValues = 1

             # While the value is the same
            while results[dim, currentId] == previousX && currentId <= size(results, 2)
                currentId += 1
                identicalValues += 1
            end

            # Add the proper points
            append!(x, previousX)
            append!(y, currentId - 1)

            if results[dim, currentId] != Inf
                append!(x, results[dim, currentId])
                append!(y, currentId - 1)
            end
            
            previousX = results[dim, currentId]
            previousY = currentId - 1
            
        end

        append!(x, maxSolveTime)
        append!(y, currentId - 1)

        # If it is the first subfolder
        if dim == 1

            # Draw a new plot
            plot(x, y, label = folderName[dim], legend = :bottomright, xaxis = "Time (s)", yaxis = "Solved instances",linewidth=3)

        # Otherwise 
        else
            # Add the new curve to the created plot
            savefig(plot!(x, y, label = folderName[dim], linewidth=3), outputFile)
        end 
    end
end 



function resultsArray(outputFile::String)
    
    resultFolder = "sol/"
    dataFolder = "data/"
    
    # Maximal number of files in a subfolder
    maxSize = 0

    # Number of subfolders
    subfolderCount = 0

    # Open the latex output file
    fout = open(outputFile, "w")

    # Print the latex file output
    println(fout, raw"""\documentclass{article}

\usepackage[french]{babel}
\usepackage [utf8] {inputenc} % utf-8 / latin1 
\usepackage{multicol}

\setlength{\hoffset}{-18pt}
\setlength{\oddsidemargin}{0pt} % Marge gauche sur pages impaires
\setlength{\evensidemargin}{9pt} % Marge gauche sur pages paires
\setlength{\marginparwidth}{54pt} % Largeur de note dans la marge
\setlength{\textwidth}{481pt} % Largeur de la zone de texte (17cm)
\setlength{\voffset}{-18pt} % Bon pour DOS
\setlength{\marginparsep}{7pt} % Séparation de la marge
\setlength{\topmargin}{0pt} % Pas de marge en haut
\setlength{\headheight}{13pt} % Haut de page
\setlength{\headsep}{10pt} % Entre le haut de page et le texte
\setlength{\footskip}{27pt} % Bas de page + séparation
\setlength{\textheight}{668pt} % Hauteur de la zone de texte (25cm)

\begin{document}""")

    header = raw"""
\begin{center}
\renewcommand{\arraystretch}{1.4} 
 \begin{tabular}{l"""

    # Name of the subfolder of the result folder (i.e, the resolution methods used)
    folderName = Array{String, 1}()

    # List of all the instances solved by at least one resolution method
    solvedInstances = Array{String, 1}()

    # For each file in the result folder
    for file in readdir(resultFolder)

        path = resultFolder * file
        
        # If it is a subfolder
        if isdir(path)

            # Add its name to the folder list
            folderName = vcat(folderName, file)
             
            subfolderCount += 1
            folderSize = size(readdir(path), 1)

            # Add all its files in the solvedInstances array
            for file2 in readdir(path)
                solvedInstances = vcat(solvedInstances, file2)
            end 

            if maxSize < folderSize
                maxSize = folderSize
            end
        end
    end

    # Only keep one string for each instance solved
    unique(solvedInstances)

    # For each resolution method, add two columns in the array
    for folder in folderName
        header *= "rr"
    end

    header *= "}\n\t\\hline\n"

    # Create the header line which contains the methods name
    for folder in folderName
        header *= " & \\multicolumn{2}{c}{\\textbf{" * folder * "}}"
    end

    header *= "\\\\\n\\textbf{Instance} "

    # Create the second header line with the content of the result columns
    for folder in folderName
        header *= " & \\textbf{Temps (s)} & \\textbf{Gap} "
    end

    header *= "\\\\\\hline\n"

    footer = raw"""\hline\end{tabular}
\end{center}

"""
    println(fout, header)

    # On each page an array will contain at most maxInstancePerPage lines with results
    maxInstancePerPage = 30
    id = 1

    # For each solved files
    for solvedInstance in solvedInstances

        # If we do not start a new array on a new page
        if rem(id, maxInstancePerPage) == 0
            println(fout, footer, "\\newpage")
            println(fout, header)
        end 

        # Replace the potential underscores '_' in file names
        print(fout, replace(solvedInstance, "_" => "\\_"))

        # For each resolution method
        for method in folderName

            path = resultFolder * method * "/" * solvedInstance

            # If the instance has been solved by this method
            if isfile(path)

                include(path)

                println(fout, " & ", round(solveTime, digits=2), " & ", round(gap,digits=3))

                
            # If the instance has not been solved by this method
            else
                println(fout, " & - & - ")
            end
        end

        println(fout, "\\\\")

        id += 1
    end

    # Print the end of the latex file
    println(fout, footer)

    println(fout, "\\end{document}")

    close(fout)
    
end 


function rewrite()

	dataFolder = "sol/"
    	resFolder = "tmp/"
#	resolutionMethod = "dual"   
#	resolutionMethod = "static"
#	resolutionMethod = "lazycut"
	resolutionMethod = "heuristic"
	dataFolder = dataFolder .* resolutionMethod
	resolutionFolder = resFolder .* resolutionMethod

	for file in readdir(dataFolder)
		datafile = open(resolutionFolder * "/" *file)

    		data = readlines(datafile)
    		close(datafile)
		
		c = split(data[2],"=")[2]
		g = split(data[3],"=")[2]
		s = split(data[length(data)],"=")[2]
		
		outputFile = resolutionFolder * "/" * file
		
		fout = open(outputFile, "w") 
		println(fout,"isOptimal = ", true)
		println(fout,"cout = ",c)
		if resolutionMethod !="heuristic"
			println(fout,"gap = ",g)
		end
		println(fout,"solveTime = ",s)

		close(fout)
	end
end