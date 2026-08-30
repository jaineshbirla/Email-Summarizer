from transformers import AutoTokenizer, pipeline

class Summarizer():
    
    def __init__(self, model_name = "sshleifer/distilbart-cnn-12-6", chunk_token_budget = 900):
        self.model_name = model_name
        self.chunk_token_budget = chunk_token_budget
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipe = pipeline("summarization", model = self.model_name, tokenizer = self.tokenizer)
        
    def _generate_summary(self, text: str, max_length : int, min_length : int) -> str:
        
        result = self.pipe(
            text,
            max_length = max_length,
            min_length = min_length,
        )
        
        return result[0]["summary_text"].strip()
    
    def _split_into_chunk(self, text: str) -> list[str]:
        
        input_ids = self.tokenizer.encode(text, add_special_tokens = False)
        
        if len(input_ids) <= self.chunk_token_budget:
            return [text]
        
        chunks = []
        
        for i in range(0, len(input_ids), self.chunk_token_budget):
            
            piece_ids = input_ids[i: i + self.chunk_token_budget]
            chunks.append(self.tokenizer.decode(piece_ids, skip_special_tokens = True))
        return chunks
    
    
    

if __name__ == "__main__":
    Summarize = Summarizer()
    print(Summarize._generate_summary("""
                                      A computer is a smart electronic device that takes raw information, processes it, and gives us useful results.Introduction to ComputersA computer is one of the greatest inventions of modern science.The word computer comes from a Latin word that means to calculate.Early computers were very large and filled whole rooms.Old machines used vacuum tubes and consumed a lot of power.Today, computers are small, fast, and fit on desks or in our hands.Modern devices include desktops, laptops, tablets, and smartphones.A standard setup has a monitor, a keyboard, a mouse, and a CPU.The Central Processing Unit (CPU) acts as the brain of the machine.Input devices let us type or click to give commands.Output devices like screens and printers show us the final work.How Computers WorkEvery computer follows a simple three-step cycle: Input, Process, Output.Input is the raw data we type or upload.Processing happens inside the CPU through logical steps.Output is the final result shown on the display.Computers process instructions using a fast binary code of zeros and ones.Memory chips store data temporarily or permanently.Hard drives and solid-state drives hold large amounts of files safely.Operating systems manage software and hardware tasks smoothly.Programs and apps give the computer specific jobs to do.Processing speeds double roughly every two years, following Moore's Law.Computers in EducationSchools use computers to make learning fun and interactive.Students read digital books and watch educational videos online.Teachers present lessons using smartboards and multimedia slides.Online classes let students study safely from their homes.The internet provides endless research material for projects.Digital exams give fast and accurate performance results.Computer labs help students learn typing and basic coding.Educational software adapts to each student's unique learning pace.Virtual science experiments make complex topics easy to see.Remote learning connects global classrooms without distance barriers.Computers in Business and WorkOffices rely on computers for daily administrative tasks.Workers create spreadsheets, reports, and professional presentations.Databases store important company and client records securely.Emails and video calls make communication fast across the world.Financial software handles payroll, taxes, and daily accounting.Automated systems reduce human error and save time.Cloud storage lets teams share files from different locations.E-commerce platforms allow businesses to sell products globally.Marketing teams use digital tools to reach target audiences.Remote work is possible because of secure network connections.""",
                                      max_length = 130 ,
                                      min_length = 30))


