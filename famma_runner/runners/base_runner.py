import json
from registrable import Registrable


class Runner(Registrable):
    @staticmethod
    def build_from_config(config):
        runner_cls = Runner.by_name(config["runner_name"].lower())
        return runner_cls(config)

    def run(self):
        """Base run method"""
        pass
    
    # @abstractmethod
    def _run_single(self, prompt: str | list[dict[str, str]]) -> list[str]:
        pass

    @staticmethod
    def run_single(combined_args) -> list[str]:
        """
        Run the model for a single prompt and return the output
        Static method to be used in multiprocessing
        Calls the _run_single method with the combined arguments
        """
        prompt: str | list[dict[str, str]]
        cache: dict[str, str]
        call_method: callable
        prompt, cache, args, call_method = combined_args

        if isinstance(prompt, list):
            prompt_cache = json.dumps(prompt)
        elif isinstance(prompt, tuple):
            prompt_cache = prompt[0] + json.dumps(prompt[1])
        else:
            prompt_cache = prompt      

        if cache is not None and prompt_cache in cache:
            if len(cache[prompt_cache]) == args.n:
                return cache[prompt_cache]

        result = call_method(prompt)
        assert len(result) == args.n

        return result

    def run_batch(self, prompts: list[str | list[dict[str, str]]]) -> list[list[str]]:
        from easyllm_kit.utils.multiprocess import run_tasks_in_parallel
        outputs = []
        arguments = [
            (
                prompt,
                self.cache,  ## pass the cache as argument for cache check
                self.args,  ## pass the args as argument for cache check
                self._run_single,  ## pass the _run_single method as argument because of multiprocessing
            )
            for prompt in prompts
        ]
        if self.args.multiprocess > 1:
            parallel_outputs = run_tasks_in_parallel(
                self.run_single,
                arguments,
                self.args.multiprocess,
                use_progress_bar=True,
            )
            for output in parallel_outputs:
                if output.is_success():
                    outputs.append(output.result)
                else:
                    print("Failed to run the model for some prompts")
                    print(output.status)
                    print(output.exception_tb)
                    outputs.extend([""] * self.args.n)
        else:
            outputs = [self.run_single(argument) for argument in arguments]

        if self.args.use_cache:
            for prompt, output in zip(prompts, outputs):
                if isinstance(prompt, list):
                    prompt_cache = json.dumps(prompt)
                elif isinstance(prompt, tuple):
                    prompt_cache = prompt[0] + json.dumps(prompt[1])
                else:
                    prompt_cache = prompt
                self.cache[prompt_cache] = output  ## save the output to cache

        return outputs
    
    