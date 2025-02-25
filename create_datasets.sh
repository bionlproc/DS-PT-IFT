#!/bin/bash

python -m process_data.CDR.process_data
python -m process_data.ChemProt.process_data
python -m process_data.DDI.process_data
python -m process_data.NaryDrugCombos.process_data