import sys
sys.path.append('.')
from app import OATSChatbot
import asyncio

async def check_actual_job_fields():
    chatbot = OATSChatbot()
    if chatbot.login():
        # Get jobs and see the actual structure
        response = await chatbot.fetch_multiple_endpoints(['jobs'], 'Show me all jobs')
        
        if 'jobs' in response:
            jobs_response = response['jobs']
            if hasattr(jobs_response, 'data') and jobs_response.data:
                if 'data' in jobs_response.data:
                    jobs_list = jobs_response.data['data']
                    if jobs_list:
                        print("=== ACTUAL JOB DATA STRUCTURE ===")
                        first_job = jobs_list[0]
                        print("First job fields and values:")
                        for field, value in first_job.items():
                            print(f"  {field}: {value}")
                        
                        print(f"\n=== ALL JOBS JOB CODES ===")
                        for i, job in enumerate(jobs_list):
                            print(f"Job {i+1} fields: {list(job.keys())}")
                            # Try different possible field names for job ID
                            possible_id_fields = ['job_code', 'id', 'job_id', 'jid', 'code']
                            job_id = None
                            for field in possible_id_fields:
                                if field in job:
                                    job_id = job[field]
                                    break
                            print(f"  Job ID (trying various fields): {job_id}")

if __name__ == "__main__":
    asyncio.run(check_actual_job_fields())
