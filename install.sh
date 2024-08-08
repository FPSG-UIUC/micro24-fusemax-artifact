python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

# Install Timeloop
cd accelergy-timeloop-infrastructure
make install_accelergy
pip3 install ./src/timeloopfe
make install_timeloop

# Install Accelergy
cd src/accelergy-library-plug-in
pip install .
cd ../../..
cp -r custom_pc_2021 env/share/accelergy/estimation_plug_ins/accelergy-library-plugin/library

jupyter lab
deactivate
