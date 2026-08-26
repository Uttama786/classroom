import random
import re
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from django.db import transaction
from flipped_app.models import StudentProfile, TeacherProfile

FIRST_NAMES = [
    # Male names
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayan", "Krishna", "Ishaan",
    "Shaurya", "Atharva", "Advait", "Pranav", "Aryan", "Dhruv", "Kabir", "Ritvik", "Darsh", "Rohan",
    "Rahul", "Amit", "Varun", "Nikhil", "Siddharth", "Harsh", "Yash", "Kunal", "Gaurav", "Mayank",
    "Ayush", "Akash", "Chirag", "Karan", "Rishabh", "Mohit", "Alok", "Dev", "Manish", "Abhishek",
    "Suresh", "Vikram", "Aniket", "Sanket", "Shubham", "Tejas", "Tanmay", "Omkar", "Prathamesh", "Pradeep",
    "Dinesh", "Manoj", "Pankaj", "Sachin", "Deepak", "Vivek", "Vishal", "Ashish", "Anand", "Rajesh",
    "Ravi", "Sanjay", "Sunil", "Ajay", "Vinay", "Hemant", "Chetan", "Tushar", "Girish", "Naveen",
    "Santosh", "Pramod", "Kiran", "Vijay", "Sandesh", "Ganesh", "Mahesh", "Ramesh", "Umesh", "Satish",
    "Bhavin", "Harshit", "Kartik", "Lalit", "Mukesh", "Nitin", "Prateek", "Rajendra", "Sumit", "Vipul",
    "Yogesh", "Brijesh", "Gautam", "Hardik", "Kailash", "Madhav", "Neeraj", "Parag", "Rakesh", "Saurabh",
    "Utkarsh", "Vaibhav", "Vikas", "Bhupendra", "Chinmay", "Devendra", "Hiten", "Jagdish", "Kapil", "Lokesh",
    # Female names
    "Ananya", "Diya", "Aadhya", "Pari", "Saanvi", "Kiara", "Myra", "Riya", "Ira", "Avani",
    "Prisha", "Riddhi", "Sneha", "Tanvi", "Anika", "Navya", "Kavya", "Ishita", "Meera", "Pooja",
    "Neha", "Swati", "Shreya", "Divya", "Simran", "Mansi", "Payal", "Sonam", "Muskan", "Kriti",
    "Richa", "Pallavi", "Chetna", "Jyoti", "Vandana", "Preeti", "Komal", "Garima", "Sakshi", "Nidhi",
    "Bhavya", "Tanisha", "Shweta", "Deepika", "Shruti", "Rashmi", "Ankita", "Akanksha", "Sunita", "Monali",
    "Pragya", "Srishti", "Ritika", "Nisha", "Meenakshi", "Shilpa", "Trisha", "Lavanya", "Charu", "Harshita",
    "Manisha", "Urvashi", "Kavita", "Suman", "Geeta", "Anjali", "Bhumika", "Chhavi", "Devanshi", "Ekta",
    "Gunjan", "Heena", "Indu", "Jaya", "Kanika", "Leena", "Madhuri", "Nandini", "Ojaswi", "Priyanka",
    "Radhika", "Saloni", "Tejaswi", "Upasana", "Vaishnavi", "Yamini", "Zoya", "Aarti", "Barkha", "Chaitali",
    "Deepa", "Esha", "Falguni", "Gauri", "Hemlata", "Isha", "Juhi", "Karishma", "Lata", "Mamta"
]

LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Patel", "Singh", "Kumar", "Rao", "Reddy", "Nair", "Iyer",
    "Joshi", "Mehta", "Shah", "Agarwal", "Mishra", "Bhat", "Deshmukh", "Kulkarni", "Patil", "Banerjee",
    "Chatterjee", "Mukherjee", "Ghosh", "Das", "Sen", "Dutta", "Roy", "Bose", "Choudhury", "Chakraborty",
    "Pillai", "Menon", "Nambiar", "Kurian", "Thomas", "Mathew", "Varghese", "Joseph", "Fernandes", "D'Souza",
    "Lobo", "Pinto", "Chauhan", "Rajput", "Rathore", "Solanki", "Tomar", "Yadav", "Maurya", "Saini",
    "Jangid", "Prajapat", "Bishnoi", "Mittal", "Bansal", "Goyal", "Singhal", "Garg", "Jindal", "Goel",
    "Mahajan", "Kapoor", "Malhotra", "Khanna", "Arora", "Sethi", "Grover", "Batra", "Anand", "Bajaj",
    "Chopra", "Dhawan", "Kohli", "Suri", "Tandon", "Ahuja", "Lamba", "Bhasin", "Duggal", "Talwar",
    "Bhatia", "Garg", "Sood", "Kaushik", "Pandey", "Tiwari", "Dubey", "Shukla", "Tripathi", "Pathak",
    "Awasthi", "Dwivedi", "Upadhyay", "Chaubey", "Vashistha", "Bhattacharya", "Sanyal", "Ganguly", "Ghoshal", "Majumdar"
]

DEPARTMENTS = ["CS", "IT", "AI", "DS", "EC", "EE", "ME", "CE"]

class Command(BaseCommand):
    help = "Replace dummy student names, usernames, emails, and roll numbers with realistic names."

    def handle(self, *args, **options):
        self.stdout.write("Fetching dummy student users...")
        dummy_users = list(
            User.objects.filter(username__startswith="student_dummy_").order_by("id")
        )
        total = len(dummy_users)
        if total == 0:
            self.stdout.write(self.style.WARNING("No dummy students found with username prefix 'student_dummy_'."))
            return

        self.stdout.write(f"Found {total} dummy students. Generating realistic names...")

        # Pre-fetch existing non-dummy usernames and emails to avoid collision
        existing_usernames = set(
            User.objects.exclude(username__startswith="student_dummy_").values_list("username", flat=True)
        )
        existing_emails = set(
            User.objects.exclude(username__startswith="student_dummy_").values_list("email", flat=True)
        )
        existing_roll_numbers = set(
            StudentProfile.objects.exclude(user__username__startswith="student_dummy_").values_list("roll_number", flat=True)
        )

        used_usernames = set(existing_usernames)
        used_emails = set(existing_emails)
        used_rolls = set(existing_roll_numbers)

        # Build realistic profile data
        user_updates = []
        profile_map = {} # user_id -> new roll_number

        rng = random.Random(42)  # Deterministic seed for reproducible nice distribution

        for idx, user in enumerate(dummy_users, start=1):
            fname = rng.choice(FIRST_NAMES)
            lname = rng.choice(LAST_NAMES)
            
            # Clean username format e.g. aarav.sharma or aarav.sharma24
            clean_first = re.sub(r'[^a-zA-Z0-9]', '', fname).lower()
            clean_last = re.sub(r'[^a-zA-Z0-9]', '', lname).lower()
            
            base_username = f"{clean_first}.{clean_last}"
            username = base_username
            suffix_num = 1
            while username in used_usernames:
                suffix_num += 1
                username = f"{base_username}{suffix_num}"
            used_usernames.add(username)

            # Email
            email = f"{username}@fliplearn.edu"
            used_emails.add(email)

            # Roll number format: 2024CS0001
            dept = DEPARTMENTS[idx % len(DEPARTMENTS)]
            year = 2023 + (idx % 3)
            roll_candidate = f"{year}{dept}{idx:04d}"
            while roll_candidate in used_rolls:
                idx += 100000
                roll_candidate = f"{year}{dept}{idx:04d}"
            used_rolls.add(roll_candidate)

            user.first_name = fname
            user.last_name = lname
            user.username = username
            user.email = email
            user_updates.append(user)
            profile_map[user.id] = roll_candidate

        self.stdout.write(f"Updating {len(user_updates)} Users in database...")
        batch_size = 1000
        with transaction.atomic():
            User.objects.bulk_update(
                user_updates,
                fields=["first_name", "last_name", "username", "email"],
                batch_size=batch_size
            )

            # Update StudentProfiles
            profiles_to_update = []
            for profile in StudentProfile.objects.filter(user_id__in=profile_map.keys()):
                profile.roll_number = profile_map[profile.user_id]
                profiles_to_update.append(profile)

            if profiles_to_update:
                self.stdout.write(f"Updating {len(profiles_to_update)} StudentProfiles in database...")
                StudentProfile.objects.bulk_update(
                    profiles_to_update,
                    fields=["roll_number"],
                    batch_size=batch_size
                )

        self.stdout.write(self.style.SUCCESS(f"Successfully converted {total} dummy students into real named students!"))
