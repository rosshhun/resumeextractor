const step1 = document.getElementById('step1');
const step2 = document.getElementById('step2');
const results = document.getElementById('results');
const loading = document.getElementById('loading');
const processResumeBtn = document.getElementById('processResume');
const processJobDescriptionBtn = document.getElementById('processJobDescription');

let resumeSkills = [];

processResumeBtn.addEventListener('click', async () => {
    const resumeText = document.getElementById('resumeText').value;
    if (!resumeText) {
        alert('Please enter your resume text.');
        return;
    }

    showLoading();

    try {
        const response = await fetch('/process_resume', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ resume_text: resumeText }),
        });

        const data = await response.json();
        resumeSkills = data.resume_skills;

        step1.classList.add('hidden');
        step2.classList.remove('hidden');
        processJobDescriptionBtn.disabled = false;
    } catch (error) {
        console.error('Error processing resume:', error);
        alert('An error occurred while processing your resume. Please try again.');
    } finally {
        hideLoading();
    }
});

processJobDescriptionBtn.addEventListener('click', async () => {
    const jobDescription = document.getElementById('jobDescription').value;
    if (!jobDescription) {
        alert('Please enter the job description.');
        return;
    }

    showLoading();

    try {
        const response = await fetch('/process_job_description', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                job_description: jobDescription,
                resume_skills: resumeSkills,
            }),
        });

        const data = await response.json();
        displayResults(data);

        step2.classList.add('hidden');
        results.classList.remove('hidden');
    } catch (error) {
        console.error('Error processing job description:', error);
        alert('An error occurred while processing the job description. Please try again.');
    } finally {
        hideLoading();
    }
});

function displayResults(data) {
    const skillScoreEl = document.getElementById('skillScore');
    const missingSkillsEl = document.getElementById('missingSkills');
    const resumeSkillsEl = document.getElementById('resumeSkills');
    const jobSkillsEl = document.getElementById('jobSkills');

    const jobSkills = data.job_skills.map(obj => obj.skill);
    const resumeSkills = data.resume_skills.map(obj => obj.skill);

    const jobSkillSet = new Set(jobSkills.map(skill => skill.toLowerCase()));
    const resumeSkillSet = new Set(resumeSkills.map(skill => skill.toLowerCase()));

    const commonSkills = new Set([...resumeSkillSet].filter(skill => jobSkillSet.has(skill)));
    const missingSkills = new Set([...jobSkillSet].filter(skill => !resumeSkillSet.has(skill)));

    const skillScore = (commonSkills.size / jobSkillSet.size) * 100;

    skillScoreEl.textContent = `Skill Match Score: ${skillScore.toFixed(2)}%`;

    missingSkillsEl.innerHTML = `
        <h3 class="text-xl font-semibold mb-2">Missing Skills:</h3>
        <ul class="list-disc pl-6">
            ${[...missingSkills].map(skill => `<li>${skill}</li>`).join('')}
        </ul>
    `;

    resumeSkillsEl.innerHTML = `
        <h3 class="text-xl font-semibold mb-2">Your Skills:</h3>
        <ul class="list-disc pl-6">
            ${resumeSkills.map(skill => `<li>${skill}</li>`).join('')}
        </ul>
    `;

    jobSkillsEl.innerHTML = `
        <h3 class="text-xl font-semibold mb-2">Job Required Skills:</h3>
        <ul class="list-disc pl-6">
            ${jobSkills.map(skill => `<li>${skill}</li>`).join('')}
        </ul>
    `;
}

function showLoading() {
    loading.classList.remove('hidden');
}

function hideLoading() {
    loading.classList.add('hidden');
}