document.getElementById('allocate_btn').onclick = async () => {
  const fileInput = document.getElementById('participant_file');
  const file = fileInput.files[0];
  if (!file) {
    alert('Please upload a CSV or XLSX file.');
    return;
  }

  const formData = new FormData();
  formData.append('file', file);
  formData.append('academic_weight', document.getElementById('academic_weight').value || '0.5');
  formData.append('normalise', document.getElementById('normalise_skill').checked);
  formData.append('ESL_weight', document.getElementById('ESL_weight').value || '0.8')
  formData.append('skill_weight', document.getElementById('skill_weight').value || '0.9')
  formData.append('min_skill', document.getElementById('min_skill').value || '0.8')
  formData.append('gender_weight', document.getElementById('gender_weight').value || '0.8')
  //solver setting
  formData.append('algorithm', document.getElementById('algorithm').value);
  formData.append('max_iterations', document.getElementById('max_iterations').value);
  formData.append('max_generations', document.getElementById('max_generations').value);
  formData.append('population_size', document.getElementById('population_size').value);
  // Output Settings
  formData.append('show_names', document.getElementById('show_names').checked);
  formData.append('show_email', document.getElementById('show_email').checked);
  formData.append('show_gender', document.getElementById('show_gender').checked);
  formData.append('show_ESL', document.getElementById('show_ESL').checked);
  formData.append('show_report', document.getElementById('show_report').checked);




  const res = await fetch('/run_allocation', {
    method: 'POST',
    body: formData
  });

  const data = await res.json();

if (res.ok) {
      console.log('Allocation result:', result);
      alert('Allocation complete! Check console or page for results.');
      // Optionally update your page with result data here
    } else {
      alert('Allocation error: ' + result.error);
    }
  } catch (error) {
    alert('Error running allocation: ' + error.message);
  }
};