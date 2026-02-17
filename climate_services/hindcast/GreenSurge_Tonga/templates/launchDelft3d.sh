#! /bin/bash 
export PATH=$PATH:/opt/intel/oneapi/mpi/2021.9.0/bin
ulimit -s unlimited
# Display usage if no arguments are provided

#I_MPI_ROOT=/software/geocean/Delft3D/2024.03/IntelMPI/mpi/2021.9.0
#I_MPI_MPIRUN=mpirun
#I_MPI_HYDRA_TOPOLIB=hwloc
#I_MPI_HYDRA_BOOTSTRAP=slurm
#I_MPI_DEBUG=5
#I_MPI_PMI_LIBRARY=/usr/lib64/slurm/mpi_pmi2.so
##########################
######   NO TOCAR   ######
##########################
case_dir=${1:-$(pwd)}
np=${2:-1}
# Specify the folder containing your model's MDU file.
mdufileFolder=$case_dir
 
# Specify the folder containing your DIMR configuration file.
dimrconfigFolder=$case_dir

# The name of the DIMR configuration file. The default name is dimr_config.xml. This file must already exist!
dimrFile=dimr_config.xml
 
# This setting might help to prevent errors due to temporary locking of NetCDF files. 
export HDF5_USE_FILE_LOCKING=FALSE

# Stop the computation after an error occurs.
set -e
 
# For parallel processes, the lines below update the <process> element in the DIMR configuration file.
# The updated list of numbered partitions is calculated from the user specified number of nodes and cores.
# You DO NOT need to modify the lines below.
PROCESSSTR="$(seq -s " " 0 $((np-1)))"
sed -i "s/\(<process.*>\)[^<>]*\(<\/process.*\)/\1$PROCESSSTR\2/" $dimrconfigFolder/$dimrFile
# The name of the MDU file is read from the DIMR configuration file.
# You DO NOT need to modify the line below.
mduFile="$(sed -n 's/\r//; s/<inputFile>\(.*\).mdu<\/inputFile>/\1/p' $dimrconfigFolder/$dimrFile)".mdu

#--- Partition by calling the dflowfm executable -------------------------------------------------------------
if [ "$np" -gt 1 ]; then 
    echo ""
    echo "Partitioning parallel model..."
    cd "$mdufileFolder"
    echo "Partitioning in folder ${PWD}"
    dflowfm --nodisplay --autostartstop --partition:ndomains="$np":icgsolver=6 "$mduFile"
    cd -
    
else 
    #--- No partitioning ---
    echo ""
    echo "Sequential model..."
fi 
 
#--- Simulation by calling the dimr executable ----------------------------------------------------------------
echo ""
echo "Simulation..."
cd $dimrconfigFolder
echo "Computing in folder ${PWD}"

#$containerFolder/execute_singularity_dimr.sh -c $containerFolder -m $modelFolder dimr "$dimrFile"
echo "mpirun -np $np dimr \"$dimrFile\""
mpirun -np $np dimr "$dimrFile"

cd -

#--- Join output files by calling the dfmoutput executable ----------------------------------------------------------------
if [ "$np" -gt 1 ]; then 
    cd $case_dir/output
    echo "Joining nc files in folder ${PWD}"
    run_dfmoutput.sh -- mapmerge --infile *map.nc
    cd -
fi