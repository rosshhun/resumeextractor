import os
import pickle
import logging
from config import OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_path):
    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
    return model


def predict_skills(model, text):
    logger.info("Predicting skills")
    predictions = model.predict([text])
    return predictions[0]


def main():
    model_path = os.path.join(OUTPUT_DIR, 'best_skill_extraction_model.pkl')
    model = load_model(model_path)

    text = """
    Sure! Here's a test resume with keywords in two-word combinations:

---

**Jane Smith**  
Email: janesmith@example.com  
Phone: (555) 987-6543  
LinkedIn: linkedin.com/in/janesmith  
GitHub: github.com/janesmith  

---

**Summary**

Experienced software engineer with a strong background in machine learning and deep learning. Adept at developing content management systems and implementing customer relationship management solutions. Proven ability to lead cross-functional teams and deliver high-quality software products.

---

**Experience**

**Senior Software Engineer**  
**Innovative Tech Solutions, Seattle, WA**  
*April 2018 – Present*

- Led the development of a machine learning model to improve recommendation systems, increasing user engagement by 25%.
- Designed and implemented a deep learning framework for image recognition, achieving 95% accuracy.
- Developed and maintained a content management system (CMS) using Python and Django.
- Integrated customer relationship management (CRM) software with existing databases to streamline operations.
- Managed a team of 8 engineers, promoting an Agile development environment.

**Software Engineer**  
**Creative Tech Solutions, Austin, TX**  
*July 2014 – March 2018*

- Developed a machine learning algorithm for fraud detection, reducing false positives by 20%.
- Implemented deep learning models for natural language processing (NLP) tasks.
- Assisted in the development of a content management system (CMS) for a large media company.
- Worked on customer relationship management (CRM) tools to enhance client interactions.
- Collaborated with UX/UI designers to create user-friendly interfaces.

---

**Education**

**Master of Science in Computer Science**  
**University of Washington, Seattle, WA**  
*September 2012 – June 2014*

**Bachelor of Science in Computer Science**  
**University of Texas, Austin, TX**  
*September 2008 – May 2012*

---

**Skills**

- **Programming Languages**: Python, Java, JavaScript, SQL
- **Frameworks and Libraries**: TensorFlow, Keras, Django, Flask, React
- **Tools and Technologies**: Git, Docker, AWS, Jenkins, JIRA
- **Databases**: MySQL, PostgreSQL, MongoDB
- **Methodologies**: Agile, Scrum, Test-Driven Development (TDD)

---

**Projects**

**Image Recognition System**  
- Developed a deep learning model for image recognition using TensorFlow and Keras.
- Achieved 95% accuracy on a dataset of 100,000 images.
- Deployed the model on AWS for scalable performance.

**Fraud Detection System**  
- Created a machine learning algorithm to detect fraudulent transactions.
- Reduced false positives by 20% through feature engineering and model optimization.
- Implemented the system in a production environment, handling over 1 million transactions per day.

---

**Certifications**

- AWS Certified Solutions Architect
- Certified ScrumMaster (CSM)
- Google Professional Machine Learning Engineer

---

**Languages**

- English (Native)
- Spanish (Conversational)

---

**Interests**

- Machine Learning and Artificial Intelligence
- Open Source Contribution
- Hiking and Outdoor Activities
- Traveling

---

This resume includes keywords such as "machine learning," "deep learning," "content management system," and "customer relationship management" to test the model's ability to recognize two-word combinations.
    """

    predictions = predict_skills(model, text)

    logger.info("Predicted skills:")
    for skill, confidence in predictions:
        logger.info(f"{skill}: {confidence:.4f}")
    print()  # Add a blank line for readability


if __name__ == "__main__":
    main()