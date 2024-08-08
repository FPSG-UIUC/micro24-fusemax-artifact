python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

# Install Timeloop
cd accelergy-timeloop-infrastructure
cd ..

# Install Accelergy
cd accelergy-library-plug-in
pip install .
cd ..
cp -r custom_pc_2021 env/share/accelergy/estimation_plug_ins/accelergy-library-plugin/library

jupyter lab
deactivate
