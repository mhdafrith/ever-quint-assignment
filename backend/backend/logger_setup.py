import logging
import logging.config
import os
import toml

def setup_logging():
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'logging_config.toml')
    log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'run_logs')
    
    # Ensure log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    if os.path.exists(config_path):
        try:
            # Python's logging.config.fileConfig expects ini/conf format usually, 
            # or dictConfig for JSON/YAML/TOML (after parsing).
            # simpler to load toml to dict and use dictConfig
            config = toml.load(config_path)
            
            # Extract loggers/handlers/formatters or map manual structure if needed
            # But python logging dictConfig schema is specific. 
            # The previous TOML looked like ini style structure.
            # Let's support the specific TOML structure provided or fallback to basic.
            
            # Actually, standard python logging dos not natively support TOML fileConfig.
            # We implemented a TOML file but need to parse it to dictConfig format.
            # INSTEAD, to strictly follow the TOML structure seen:
            # [logger.root] -> handlers
            
            # Let's map it manually to a dictConfig for safety
            
            logging_config = {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "simple": {
                        "format": config['formatter']['simple']['format']
                    }
                },
                "handlers": {
                    "console": {
                        "class": "logging.StreamHandler",
                        "level": "INFO",
                        "formatter": "simple",
                        "stream": "ext://sys.stdout"
                    },
                    "file": {
                        "class": "logging.FileHandler",
                        "level": "INFO",
                        "formatter": "simple",
                        "filename": os.path.join(log_dir, "app.log"),
                        "mode": "a"
                    }
                },
                "root": {
                    "level": "INFO",
                    "handlers": ["console", "file"]
                }
            }
            logging.config.dictConfig(logging_config)
            logging.info("Logging configured from toml via dictConfig.")
        except Exception as e:
            print(f"Error loading logging config: {e}")
            logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)
