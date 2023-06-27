import os




env_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

dir_env_path = os.path.abspath(os.path.dirname(env_path))

model_path = os.path.join(dir_env_path,'models')

yaml_path = os.path.join(env_path,'yaml')

log_path = os.path.join(env_path,'log')

main_args_path = os.path.join(yaml_path,f"main.yml" )


if __name__ == '__main__':
    constant = [env_path,yaml_path,model_path]
    for i in constant:
        print(i)