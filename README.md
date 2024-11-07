# Trash-Classification-ML-Project

# How to run?

## **Step 1 Clone the repository in your working directory**
```bash
git clone https://github.com/AnggaDS01/Trash-Classification-ML-Project.git
```

```bash
cd Trash-Classification-ML-Project
```

## **Step 2 Create a virtual environment after opening the repository**
Here I am using python version 3.9.2, in creating a virtual environment you can use [Anaconda](https://www.anaconda.com/download/success) or [Miniconda](https://docs.anaconda.com/miniconda/), here I use Miniconda version 24.9.2, to create it you can type the following command:

### **Step 2.1 Create a virtual environment using the conda prompt**
```bash
# conda create -n <directory_name> python=<python_version> ipython
conda create -n "myenv" python=3.9.2 ipython
```

#### **Step 2.1.1 Activate the virtual environment**

```bash
# To activate this environment, use
#
#     $ conda activate myenv
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

when you attempt to type the command `conda activate myenv`

```bash 
conda activate myenv
```

and get an error like this:

Output:
```bash
CondaError: Run 'conda init' before 'conda activate'
```

> **Note: Please restart your shell and open it again in your shell, then your conda will be activated.**

### **Step 2.2 Create a virtual environment using the venv module**
```bash
# python -m venv <directory>
python -m venv myenv
```

#### **Step 2.2.1 Windows venv activation**
To activate your venv on Windows, you need to run a script that gets installed by venv. If you created your venv in a directory called `myvenv`, the command would be:

```bash
# In cmd.exe
myenv\Scripts\activate.bat
# In PowerShell
myenv\Scripts\Activate.ps1
```

#### **Step 2.2.2 Linux and MacOS venv activation**
On Linux and MacOS, we activate our virtual environment with the source command. If you created your venv in the `myvenv` directory, the command would be:

```bash
$ source myvenv/bin/activate
```

## **Step 3 Install the requirements**

```bash
pip install -r requirements.txt
```

## **Step 4 Run the application**

```bash
python app.py
```