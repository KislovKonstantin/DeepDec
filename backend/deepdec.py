import requests # отправка запросов через API
import json # для считывания CONFIG и сохранения результата работы системы
import re # парсинг из строк
import concurrent.futures # параллельность
import time # измерение времени
from abc import ABC, abstractmethod # абстрактные класс и метод
from pathlib import Path # ООП обертка для пути к файлу
import zmq # для отправки отладочных сообщений в сокет (прослушивается фронтендом)
from typing import Optional, List, Dict, Any # ООП обертка типов
import openai # доступ к API OpenAI
from enum import Enum # перечисления
from dataclasses import dataclass # для заморозки класса данных
import logging # логирование

# Настройка логирования
logging.basicConfig(filename='dd_log_file.log',level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',filemode='w')

# Класс для отправки сообщений о прогрессе выполнения обработки в сокет, которые считывает фронт
class SocketMessenger:
    def __init__(self, socket_addr: str = "tcp://localhost:5555"):
        self.socket_addr = socket_addr
        self.context: Optional[zmq.Context] = None
        self.socket: Optional[zmq.Socket] = None
        self._initialize_socket()

    def _initialize_socket(self):
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB) # независимый издатель
            self.socket.connect(self.socket_addr)
            self.send("System", "ZeroMQ connection established") # пока по сообщению нельзя понять, какой именно издатель подключен успешно (можно улучшить)
        except Exception as e:
            logging.error(f"ZeroMQ Error: {str(e)}") # пока по сообщению нельзя понять, какому именно издателю не удалось подключиться (можно улучшить)
            self.context = None
            self.socket = None

    def send(self, sender: str, message: str) -> None:
        try:
            if self.socket:
                self.socket.send_string(f"{sender}: {message}")
                logging.info(f"{sender}: {message}")

        except Exception as e:
            logging.error(f"ZeroMQ Send Error: {str(e)}") # пока по сообщению нельзя понять, какому именно издателю не удалось подключиться (можно улучшить)

# Класс для хранения данных API и их валидации (в зачаточном виде, можно улучшить)
@dataclass(frozen=True) # замораживаем, так как это ключи к словарю CONFIG
class APIConfig:
    api_type: str # openrouter / openai / huggingface
    api_key: str
    api_url: Optional[str] = None

    def validate(self):
        if not self.api_type or not self.api_key:
            raise ValueError("API type and API key must be specified.")


# Класс для хранения данных модели нейронки и их валидации (в зачаточном виде, можно улучшить)
@dataclass(frozen=True) # замораживаем, так как это ключ к словарю CONFIG
class ModelConfig:
    model_name: str

    def validate(self):
        if not self.model_name:
            raise ValueError("Model name must be specified.")

# Класс для хранения типов событий (для агентов и веток агентов)
class SystemEvents(Enum):
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"

# Класс события
class Event:
    def __init__(self, name: SystemEvents, data: Dict[str, Any]):
        self.name = name
        self.data = data

# Абстрактный класс наблюдателя
class Observer(ABC):
    @abstractmethod
    def update(self, event: Event):
        pass

# Логирование действий агентов
class AgentObserver(Observer):
    def update(self, event: Event):
        logging.info(f"Agent Event: {event.name}, Data: {event.data}")

# Логирование действий веток агентов
class WorkflowObserver(Observer):
    def update(self, event: Event):
        logging.info(f"Workflow Event: {event.name}, Data: {event.data}")

# Простая система рассылки событий подписанным наблюдателям
class EventSystem:
    def __init__(self):
        self.observers: List[Observer] = []

    def register(self, observer: Observer):
        self.observers.append(observer)

    def unregister(self, observer: Observer):
        self.observers.remove(observer)

    def post(self, event: Event):
        for observer in self.observers:
            observer.update(event)

# Особый наблюдатель для оркестратора
class SystemOrchestratorObserver(Observer):
    def __init__(self, system_orchestrator):
        self.system_orchestrator = system_orchestrator

    def update(self, event: Event):
        self.system_orchestrator.handle_event(event) # реализация логирования в классе SystemOrchestrator

# Выделим событие - результат работы агента (тоже логируется)
class AgentResultEvent(Event):
    def __init__(self, agent_name: str, result: str):
        super().__init__(SystemEvents.AGENT_COMPLETED, {"agent_name": agent_name, "result": result})

# Абстрактный класс обработки ошибок
class ErrorHandler(ABC):
    @abstractmethod
    def handle_error(self, error: Exception, context: Dict[str, Any]):
        pass

    @abstractmethod
    def set_next(self, handler: "ErrorHandler"): # для создания цепочки обработчиков ошибок
        pass

# Абстрактная цепочка обработчиков ошибок
class BaseErrorHandler(ErrorHandler):
    _next_handler: ErrorHandler = None

    def set_next(self, handler: ErrorHandler):
        self._next_handler = handler
        return handler

    @abstractmethod
    def handle_error(self, error: Exception, context: Dict[str, Any]): # переход к следующему обработчику
        if self._next_handler:
            return self._next_handler.handle_error(error, context)
        return None

# Обработка ошибок API
class APIErrorHandler(BaseErrorHandler):
    def handle_error(self, error: Exception, context: Dict[str, Any]):
        if isinstance(error, requests.exceptions.RequestException):
            message = f"API Request Error: {str(error)}"
            logging.error(message)
            return message
        return super().handle_error(error, context)

# Обработка ошибок считывания CONFIG
class JSONDecodeErrorHandler(BaseErrorHandler):
    def handle_error(self, error: Exception, context: Dict[str, Any]):
        if isinstance(error, json.JSONDecodeError):
            message = f"JSON Decode Error: {str(error)}"
            logging.error(message)
            return message
        return super().handle_error(error, context)

# Обработка ошибки отсутствия файла (CONFIG)
class FileNotFoundErrorHandler(BaseErrorHandler):
    def handle_error(self, error: Exception, context: Dict[str, Any]):
        if isinstance(error, FileNotFoundError):
            config_file = context.get("config_file", "unknown")
            message = f"CONFIG Is Not Found Error: {config_file}"
            logging.error(message)
            return message
        else:
            return super().handle_error(error, context)

# Обработка иных ошибок
class GenericErrorHandler(BaseErrorHandler):
    def handle_error(self, error: Exception, context: Dict[str, Any]):
        message = f"Generic Error: {str(error)}"
        logging.error(message)
        return message

# Абстрактный класс API
class APIClient(ABC):
    def __init__(self, config: APIConfig):
        self.config = config
        self.messenger = SocketMessenger()
        self.config.validate()

    @abstractmethod
    def send_request(self, model: str, prompt: str):
        pass

# Фасад для различных API
class APIGateway:
    def __init__(self, config: Dict):
        self.config = config
        self.api_client_factory = APIClientFactory() # фабрика API (ниже)
        self.api_client = self.api_client_factory.create_api_client(config) # конкретный API
        self.error_handler = APIErrorHandler()
        self.error_handler.set_next(JSONDecodeErrorHandler()).set_next(GenericErrorHandler())
        self.messenger = SocketMessenger()
    def send_request(self, model: str, prompt: str):
        try:
            return self.api_client.send_request(model, prompt)
        except Exception as e:
            logging.error(f"API Request failed: {str(e)}")
            error_message = self.error_handler.handle_error(e, {"model": model, "prompt": prompt})
            self.messenger.send("API", error_message or "Unknown API Error")
            return None

# Фабрика API (обработка типа API)
class APIClientFactory:
    def create_api_client(self, config: Dict):
        api_config = APIConfig(**config)
        if api_config.api_type == "openrouter":
            return OpenRouterAPIClient(api_config)
        elif api_config.api_type == "openai":
            return OpenAIAPIClient(api_config)
        elif api_config.api_type == "huggingface":
            return HuggingFaceAPIClient(api_config)
        else:
            error_msg = f"Unsupported API type: {api_config.api_type}"
            logging.error(error_msg)
            return None


# Бесплатный тестовый API (OpenRouter)
class OpenRouterAPIClient(APIClient):
    def __init__(self, config: APIConfig):
        super().__init__(config)
        self.headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def send_request(self, model: str, prompt: str):
        try:
            response = requests.post(
                url=self.base_url,
                headers=self.headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=60
            )
            response.raise_for_status()
            response_text = response.json()["choices"][0]["message"]["content"]
            logging.info(f"Intermediate response: {response_text}") # для отладки логируем все ответы (можно убрать)
            return response_text

        except Exception as e:
            self.messenger.send("API", f"OpenRouter API Error: {str(e)}")
            return None

# OpenAI
class OpenAIAPIClient(APIClient):
    def __init__(self, config: APIConfig):
        super().__init__(config)
        self.client = openai.OpenAI(api_key=self.config.api_key)

    def send_request(self, model: str, prompt: str):
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            response_text = completion.choices[0].message.content
            logging.info(f"Intermediate response: {response_text}")  # для отладки логируем все ответы (можно убрать)
            return response_text
        except Exception as e:
            self.messenger.send("API", f"OpenAI API Error: {str(e)}")
            return None

# HuggingFace
class HuggingFaceAPIClient(APIClient):
    def __init__(self, config: APIConfig):
        super().__init__(config)
        self.headers = {"Authorization": f"Bearer {self.config.api_key}"}

    def query(self, payload):
        try:
            response = requests.post(self.config.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.messenger.send("API", f"Hugging Face API Request Error: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            self.messenger.send("API", f"Hugging Face API JSON Decode Error: {str(e)}")
            return None

    def send_request(self, model: str, prompt: str):
        try:
            response = self.query({"inputs": prompt})
            if isinstance(response, list) and len(response) > 0:
                response_text = response[0].get('generated_text', None)
                if response_text:
                    logging.info(
                        f"Intermediate response: {response_text}")  # для отладки логируем все ответы (можно убрать)
                    return response_text
                else:
                    self.messenger.send("API", "Hugging Face API: No text generated in response.")
                    return None
            else:
                self.messenger.send("API", f"Hugging Face API: Unexpected response format: {response}")
                return None
        except Exception as e:
            self.messenger.send("API", f"Hugging Face API Error: {str(e)}")
            return None

# Парсинг ответов/оценок/комментариев
class ResponseParser:
    @staticmethod
    def extract_answer(response: str):
        answer_match = re.search(r'\[ANSWER\](.*?)(?=\n\[|\Z)', response, re.DOTALL)
        return answer_match.group(1).strip() if answer_match else response

    @staticmethod
    def extract_mark(response: str):
        mark_match = re.search(r'\[MARK\]\s*(YES|NO)', response, re.IGNORECASE)
        return mark_match.group(1).upper() if mark_match else "NO"

    @staticmethod
    def extract_notes(response: str, mark: str):
        if mark == "NO":
            notes_match = re.search(r'\[REVISION NOTES\](.*?)(?=\n\[MARK\]|\Z)', response, re.DOTALL)
            return notes_match.group(1).strip() if notes_match else "General improvement needed"
        return ""

# Состояние агента
class AgentState(ABC):
    @abstractmethod
    def generate_response(self, agent: "BaseAgent", context: str, feedback: str = ""):
        pass

# Готовый агент, может генерировать ответ
class ReadyAgentState(AgentState):
    def generate_response(self, agent: "BaseAgent", context: str, feedback: str = ""):
        full_prompt = agent.base_prompt + f"\n{context}"
        if feedback:
            full_prompt += f"\n[PREVIOUS ERRORS - DO NOT REPEAT]\n{feedback}"
        agent.full_prompt = full_prompt
        agent.raw_response = agent.api_gateway.send_request(agent.model_config.model_name, agent.full_prompt)
        return agent.raw_response

# Общий класс агентов
class BaseAgent(ABC):
    def __init__(self, api_gateway: APIGateway, model_config: ModelConfig,prompt_path: Path,name: str,event_system: EventSystem):
        self.api_gateway = api_gateway
        self.model_config = model_config
        self.model_config.validate()
        self.name = name
        self.event_system = event_system
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.base_prompt = f.read()
        self.full_prompt = ""
        self.raw_response = ""
        self.analyzed_response = ""
        self.previous_errors = ""
        self.state: AgentState = ReadyAgentState() # агент после инициализации готов работать

    def generate_response(self, context: str, feedback: str = ""):
        return self.state.generate_response(self, context, feedback)

# Аналитик / Комментатор / Реконструктор
class CodeAgent(BaseAgent):
    def __init__(self, api_gateway: APIGateway, model_config: ModelConfig, prompt_path: Path, name: str, event_system: EventSystem):
      super().__init__(api_gateway, model_config, prompt_path, name, event_system)
    pass

# Оценщик
class EvaluatorAgent(BaseAgent):
    def __init__(self, api_gateway: APIGateway, model_config: ModelConfig, prompt_path: Path, name: str, event_system: EventSystem):
      super().__init__(api_gateway, model_config, prompt_path, name, event_system)
    def generate_response(self, agent_prompt: str, agent_response: str):
        full_prompt = self.base_prompt.replace("[PROMPT]", agent_prompt).replace("[RESPONSE]", agent_response)
        self.full_prompt = full_prompt
        self.raw_response = self.api_gateway.send_request(self.model_config.model_name, self.full_prompt)
        return self.raw_response

# Агрегатор
class AggregatorAgent(BaseAgent):
    def __init__(self, api_gateway: APIGateway, model_config: ModelConfig, prompt_path: Path, name: str, event_system: EventSystem):
      super().__init__(api_gateway, model_config, prompt_path, name, event_system)
    pass

# Собираем агента, оценщика, парсера, систему событий в одном месте
class WorkflowAggregate:
    def __init__(self, name: str, agent: BaseAgent, evaluator: BaseAgent, parser: ResponseParser, event_system: EventSystem):
        self.name = name
        self.agent = agent
        self.evaluator = evaluator
        self.parser = parser
        self.event_system = event_system

# Абстрактная ветка (агент+оценщик)
class AsyncWorkflow(ABC):
    def __init__(self, workflow_aggregate: WorkflowAggregate, max_attempts: int = 3):
        self.messenger = SocketMessenger()
        self.workflow_aggregate = workflow_aggregate
        self.max_attempts = max_attempts
        self.event_system = workflow_aggregate.event_system

    def _execute_workflow(self, context: str, feedback_history: list):
        agent_name = self.workflow_aggregate.agent.name
        workflow_name = self.workflow_aggregate.name
        final_response = None
        for attempt in range(self.max_attempts):
            start_time = time.time()
            response = self.workflow_aggregate.agent.generate_response(context, "\n\n".join(feedback_history))
            end_time = time.time()
            self.messenger.send(agent_name,f"[{agent_name}] Attempt {attempt + 1}: Agent response generated in {end_time - start_time:.2f} seconds")
            if not response:
                self.messenger.send(agent_name,f"[{agent_name}] Attempt {attempt + 1}: Agent response was empty, skipping evaluation.")
                continue
            final_response = response
            self.workflow_aggregate.agent.analyzed_response = self._process_response(response)
            eval_response = self._evaluate_response(attempt)
            if not eval_response:
                continue
            mark = self.workflow_aggregate.parser.extract_mark(eval_response)
            if mark == "YES":
                self.messenger.send(agent_name, f"[{agent_name}] Attempt {attempt + 1}: Mark is YES. Response is sending to Aggregator")
                self._handle_success(workflow_name, agent_name)
                return self.workflow_aggregate.agent.analyzed_response
            feedback = self.workflow_aggregate.parser.extract_notes(eval_response, mark)
            feedback_history.append(feedback)
            self.workflow_aggregate.agent.previous_errors = f"\n[PREVIOUS ERRORS - DO NOT REPEAT]\n{feedback}"
            self.messenger.send(agent_name, f"[{agent_name}] Attempt {attempt + 1}: Mark is NO. Feedback received.")
        self._handle_failure(workflow_name, agent_name, final_response)
        return final_response

    @abstractmethod
    def _process_response(self, response: str):
        pass

    @abstractmethod
    def _evaluate_response(self, attempt: int):
        pass

    @abstractmethod
    def _handle_success(self, workflow_name: str, agent_name: str):
        pass

    def _handle_failure(self, workflow_name: str, agent_name: str, final_response: Optional[str]):
        self.event_system.post(Event(SystemEvents.WORKFLOW_FAILED, {"workflow_name": workflow_name}))
        self.messenger.send(agent_name, f"[{agent_name}] Workflow failed after {self.max_attempts} attempts.")
        self.event_system.post(AgentResultEvent(agent_name, final_response or "No valid output"))

# Ветка с аналитиком / комментатором / реконструктором и оценщиком
class CodeWorkflow(AsyncWorkflow):
    def run(self, code: str):
        self.event_system.post(Event(SystemEvents.WORKFLOW_STARTED, {"workflow_name": self.workflow_aggregate.name}))
        try:
            result = self._execute_workflow(code, [])
            return result
        except Exception as e:
            self._handle_exception(e)

    def _process_response(self, response: str):
        return self.workflow_aggregate.parser.extract_answer(response)

    def _evaluate_response(self, attempt: int):
        eval_prompt = self.workflow_aggregate.agent.full_prompt
        eval_response = self.workflow_aggregate.evaluator.generate_response(eval_prompt, self.workflow_aggregate.agent.analyzed_response)
        return eval_response

    def _handle_success(self, workflow_name: str, agent_name: str):
        self.event_system.post(AgentResultEvent(agent_name, self.workflow_aggregate.agent.analyzed_response))
        self.event_system.post(Event(SystemEvents.WORKFLOW_COMPLETED, {"workflow_name": workflow_name}))

    def _handle_exception(self, e: Exception):
        agent_name = self.workflow_aggregate.agent.name
        self.messenger.send(agent_name, f"[{agent_name}] Workflow failed with exception: {str(e)}")
        self.event_system.post(Event(SystemEvents.WORKFLOW_FAILED, {"workflow_name": self.workflow_aggregate.name}))

# Ветка агрегатора и оценщика
class AggregatorWorkflow(AsyncWorkflow):
    def run(self, analyst_response: str, comment_response: str, reconstruct_response: str):
        self.event_system.post(Event(SystemEvents.WORKFLOW_STARTED, {"workflow_name": self.workflow_aggregate.name}))
        try:
            full_context = f"[ANALYST]\n{analyst_response}\n\n[COMMENTATOR]\n{comment_response}\n\n[RECONSTRUCTOR]\n{reconstruct_response}"
            result = self._execute_workflow(full_context, [])
            return result
        except Exception as e:
            self._handle_exception(e)

    def _process_response(self, response: str):
        return response

    def _evaluate_response(self, attempt: int):
        return self.workflow_aggregate.evaluator.generate_response(
            self.workflow_aggregate.agent.base_prompt,
            self.workflow_aggregate.agent.analyzed_response
        )

    def _handle_success(self, workflow_name: str, agent_name: str):
        self.event_system.post(AgentResultEvent(agent_name, self.workflow_aggregate.agent.analyzed_response))
        self.event_system.post(Event(SystemEvents.WORKFLOW_COMPLETED, {"workflow_name": workflow_name}))

    def _handle_exception(self, e: Exception):
        agent_name = self.workflow_aggregate.agent.name
        self.messenger.send(agent_name, f"[{agent_name}] Workflow failed with exception: {str(e)}")
        self.event_system.post(Event(SystemEvents.WORKFLOW_FAILED, {"workflow_name": self.workflow_aggregate.name}))

# Запускаем и управляем всей системой в оркестраторе
class SystemOrchestrator:
    # Внутри построим весь оркестратор пошагово
    class Builder:
        def __init__(self, config: Dict):
            self.config = config
            self.event_system = EventSystem()
            self.observer = AgentObserver()
            self.api_gateway = APIGateway(config["api"])
            self.workflows: Dict[str, WorkflowAggregate] = {}

        def with_agent_observer(self, observer: AgentObserver):
            self.observer = observer
            self.event_system.register(observer)
            return self

        def with_workflow_observer(self, observer: WorkflowObserver):
            self.event_system.register(observer)
            return self

        def with_workflow(self, name: str, agent_type: str, eval_type: str):
            agent = self._create_agent(agent_type, name)
            evaluator = self._create_evaluator(eval_type, name)
            parser = ResponseParser()
            self.workflows[name] = WorkflowAggregate(name, agent, evaluator, parser, self.event_system)
            return self

        def _create_agent(self, agent_type: str, name: str):
            api_gateway = self.api_gateway
            model_config = ModelConfig(model_name=self.config["models"][agent_type])
            prompt_path = Path(self.config["prompts"][agent_type])
            event_system = self.event_system
            if agent_type == "analysis" or agent_type == "commentary" or agent_type == "reconstruct":
                return CodeAgent(api_gateway, model_config, prompt_path, name, event_system)
            elif agent_type == "aggregator":
                return AggregatorAgent(api_gateway, model_config, prompt_path, name, event_system)
            else:
                raise ValueError(f"Unsupported agent type: {agent_type}")

        def _create_evaluator(self, eval_type: str, name: str):
            api_gateway = self.api_gateway
            model_config = ModelConfig(model_name=self.config["models"][eval_type])
            prompt_path = Path(self.config["prompts"][eval_type])
            event_system = self.event_system
            return EvaluatorAgent(api_gateway, model_config, prompt_path, name, event_system)

        def build(self):
            return SystemOrchestrator(self)

    def __init__(self, builder: Builder):
        self.messenger = SocketMessenger()
        self.config = builder.config
        self.api_gateway = builder.api_gateway
        self.results = {}
        self.workflows = {
            name: CodeWorkflow(workflow, max_attempts=3) if name != "aggregator"
            else AggregatorWorkflow(workflow, max_attempts=3)
            for name, workflow in builder.workflows.items()
        }
        self.event_system = builder.event_system
        self.event_system.register(SystemOrchestratorObserver(self))

    def handle_event(self, event: Event):
        logging.info(f"System Orchestrator received event: {event.name}, Data: {event.data}")

    def _load_code(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def execute(self, input_file: str, output_file: str):
        if self.api_gateway.api_client==None:
            self.messenger.send('API', f"Unsupported API type: please, restart program")
            return
        code = self._load_code(input_file)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(wf.run, code): name
                for name, wf in self.workflows.items()
                if name != "aggregator" # агрегатор запускается после всех остальных веток агентов
            }
            for future in concurrent.futures.as_completed(futures):
                branch = futures[future]
                try:
                    self.results[branch] = self.workflows[branch].workflow_aggregate.agent.analyzed_response or "No valid output"
                except Exception as e:
                    self.messenger.send(branch, f"Error in {branch}: {str(e)}")
                    self.results[branch] = "Processing failed"
        analyst_result = self.results.get("analysis", "")
        comment_result = self.results.get("commentary", "")
        reconstruct_result = self.results.get("reconstruction", "")
        agg_workflow = self.workflows["aggregator"]
        agg_result = agg_workflow.run(analyst_result, comment_result, reconstruct_result)
        self.results["aggregated"] = agg_result or "Aggregation failed"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=4)
        self.messenger.send("System", f"Results saved to {output_file}")
        self.messenger.send("System", f"System finished")

# Загрузка конфига (можно завернуть в статический метод)
def load_config(config_file: str):
    error_handler = FileNotFoundErrorHandler()
    error_handler.set_next(GenericErrorHandler())
    context = {"config_file": config_file}
    try:
        with open(config_file, "r") as f:
            config_data = eval(f.read())
            return config_data
    except Exception as e:
        error_message = error_handler.handle_error(e, context)
        return None

if __name__ == "__main__":
    CONFIG_FILE = "config.json" # лежит в директории с фронтом
    INPUT_FILE = "test.txt" # лежит в директории с фронтом
    OUTPUT_FILE = "full.json"
    CONFIG = load_config(CONFIG_FILE)
    # все промпты тоже лежат в папке с фронтом
    if CONFIG:
        orchestrator = SystemOrchestrator.Builder(CONFIG) \
            .with_agent_observer(AgentObserver()) \
            .with_workflow_observer(WorkflowObserver()) \
            .with_workflow("analysis", "analysis", "eval_analysis") \
            .with_workflow("commentary", "commentary", "eval_comment") \
            .with_workflow("reconstruction", "reconstruct", "eval_reconstruct") \
            .with_workflow("aggregator", "aggregator", "eval_aggregator") \
            .build()
        orchestrator.execute(INPUT_FILE, OUTPUT_FILE)
    else:
        print("CONFIG Error")