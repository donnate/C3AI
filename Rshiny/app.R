# Define UI for app that draws a histogram ----
library("shiny")
library("tidyverse")
source("preprocessing.R")

VIRTUALENV_NAME = 'c3ai_env'

countries = read.csv("countries_names.csv")
countries_list = list()
for (c in unique(countries$country_name)){
  if (sum(countries$country_name == c ) >1 ){
    countries_list[[c]]= as.list(sapply(countries$region[countries$country_name == c], function(x){toString(x)}))
  }else{
    countries_list[[c]]= list(c("Main territory"))
  }
}
COVID_report = read.csv("Report_data.csv", header=T)
PYTHON_DEPENDENCIES = c('numpy', 'setuptools==49.3.0', 'scipy', 'pandas==0.23.3', 'requests')
# ------------------------- Settings (Do not edit) -------------------------- #


if (Sys.info()[['user']] == 'shiny'){
  # Running on shinyapps.io
  Sys.setenv(PYTHON_PATH = 'python3')
  Sys.setenv(VIRTUALENV_NAME = VIRTUALENV_NAME)# Installs into default shiny virtualenvs dir
  Sys.setenv(RETICULATE_PYTHON = paste0('/home/shiny/.virtualenvs/', VIRTUALENV_NAME, '/bin/python'))
} else if (Sys.info()[['user']] == 'rstudio-connect'){
  
  # Running on remote server
  Sys.setenv(PYTHON_PATH = '/opt/python/3.7.6/bin/python')
  Sys.setenv(VIRTUALENV_NAME = paste0(VIRTUALENV_NAME, '/')) # include '/' => installs into rstudio-connect/apps/
  Sys.setenv(RETICULATE_PYTHON = paste0(VIRTUALENV_NAME, '/bin/python'))
  
} else {
  # Running locally
  PYTHON_DEPENDENCIES = c('numpy', 'pandas', 'scipy','scikit-learn', 'requests')
  options(shiny.port = 7450)
  Sys.setenv(PYTHON_PATH = 'python3')
  Sys.setenv(VIRTUALENV_NAME = VIRTUALENV_NAME) # exclude '/' => installs into ~/.virtualenvs/
}

virtualenv_dir = Sys.getenv('VIRTUALENV_NAME')
print(virtualenv_dir)
python_path = Sys.getenv('PYTHON_PATH')
print(python_path)


reticulate::virtualenv_create(envname = virtualenv_dir, python = python_path)
reticulate::virtualenv_install(virtualenv_dir, packages = PYTHON_DEPENDENCIES, ignore_installed = TRUE)
reticulate::use_virtualenv(virtualenv_dir, required = T)
print(reticulate::py_discover_config())



ui <- fluidPage(
  
  # App title ----
  titlePanel("What is the probability that you have COVID?"),
  
  # Sidebar layout with input and output definitions ----
  sidebarLayout(
    
    # Sidebar panel for inputs ----
    sidebarPanel(
      
      # Input: Slider for the number of bins ----
      #radioButtons(inputId = "test",
      #            label = "Have you taken an antibody test before?",
      #             choices =  c( "No"= 0,"Yes" = 1), inline = TRUE),
      radioButtons(inputId = "test",
                   label = "If you have taken a COVID test, what was your result?",
                   choices =  c(  "Negative"= 0,
                                  "Positive" = 1,
                                  "I haven't taken any tests"= 2),
                   selected = 2,
                   inline = FALSE),
      radioButtons(inputId = "type_test",
                   label = "What kind of test was it?",
                   choices =  c(  "PCR"= "pcr",
                                  "At home/Rapid test" = "rapid",
                                  "I haven't taken any tests"= "missing"),
                   selected = "missing",
                   inline = FALSE),
      
      radioButtons(inputId = "symptomatic",
                   label="Have you felt ill/ any symptoms in the days leading to your test?",
                   choices = c("Yes" = 1,
                               "No" = 0),
                   selected = 0,
                   inline = TRUE),
      conditionalPanel(
        condition = "input.test != 2",
        dateInput(inputId = "date_test",
                  label ="When did you take the test?", 
                  value ="2020-09-01",
                  min = NULL,
                  max = NULL,
                  format = "yyyy-mm-dd",
                  startview = "month",
                  weekstart = 0,
                  language = "en",
                  width = NULL,
                  autoclose = TRUE,
                  datesdisabled = NULL,
                  daysofweekdisabled = NULL)
      ),
      conditionalPanel(
        condition = "input.test != 2",
        numericInput(inputId = "symptom_onset",
                   label = "How long after your symptoms started did you take the test?",
                   min = 0,
                   max= 60,
                   value=4
                   )),
      conditionalPanel(
        condition = "input.symptomatic == 1",
        radioButtons(inputId = "shortnessOfBreath",
                   label = "Shortness Of Breath?",
                   choices =  c( "No"= 0,"Yes" = 1),selected =0, inline = TRUE)),
      conditionalPanel(
        condition = "input.symptomatic == 1",
        radioButtons(inputId = "myalgia",
                   label = "Achy Joints or Muscles?",
                   choices =  c( "No"= 0,"Yes" = 1),selected =0, inline = TRUE)),
      conditionalPanel(
        condition = "input.symptomatic == 1",
        radioButtons(inputId = "fever",
                   label = "A fever/ high temperature?",
                   choices =c( "No"= 0,"Yes" = 1),selected =0, inline = TRUE)),
      conditionalPanel(
        condition = "input.symptomatic == 1",
        radioButtons(inputId = "headache",
                   label = "Headaches?",
                   choices = c( "No"= 0,"Yes" = 1), selected =0,inline = TRUE)),
      conditionalPanel(
        condition = "input.symptomatic == 1",
        radioButtons(inputId = "runny_nose",
                   label = "A runny nose?",
                   choices =c( "No"= 0,"Yes" = 1),selected =0, inline = TRUE)),
      conditionalPanel(
        condition = "input.symptomatic == 1",
        radioButtons(inputId = "soreThroat",
                   label = "A sore throat?",
                   choices = c( "No"= 0,"Yes" = 1),selected =0, inline = TRUE)),
      conditionalPanel(
        condition = "input.symptomatic == 1",
        radioButtons(inputId = "lossTaste",
                   label = "Any loss of taste?",
                   choices =  c( "No"= 0,"Yes" = 1),selected =0, inline = TRUE)),
      conditionalPanel(
        condition = "input.symptomatic == 1",
        radioButtons(inputId = "lossSmell",
                     label = "Any loss of smell?",
                     choices =  c( "No"= 0,"Yes" = 1), selected =0, inline = TRUE)),
      conditionalPanel(
        condition = "input.symptomatic == 1",
        radioButtons(inputId = "Cough",
                   label = "A cough?",
                   choices = c( "No"= 0,"Yes" = 1), selected =0, inline = TRUE)),
      conditionalPanel(
        condition = "input.symptomatic == 1",
        radioButtons(inputId = "fatigue",
                   label = "Fatigue?",
                   choices =  c( "No"= 0,"Yes" = 1), selected =0,inline = TRUE)),
      conditionalPanel(
        condition = "input.symptomatic == 1",
        radioButtons(inputId = "stomachUpsetDiarrhoea",
                   label = " An upset stomach/ diarrhoea?",
                   choices =  c( "No"= 0,"Yes" = 1),selected =0, inline = TRUE)),
      conditionalPanel(
        condition = "input.symptomatic == 1 & input.sob == 1",
        sliderInput(inputId = "howShortOfBreath",
                  label = "How short of breath did/do you feel?",
                  min = 0,
                  max = 3,
                  value = 0)),
      conditionalPanel(
      condition = "input.symptomatic == 1 & input.fever == 1",
      sliderInput(inputId = "fever_severity",
                  label = "How bad was your fever?",
                  min = 0,
                  max = 3,
                  value = 0)),
      conditionalPanel(
      condition = "input.symptomatic == 1 & input.Cough == 1",
      sliderInput(inputId = "cough_severity",
                  label = "How bad was your cough?",
                  min = 0,
                  max = 3,
                  value = 0)),
      sliderInput(inputId = "numberInHousehold",
                  label = "How many other people do you live with?",
                  min = 0,
                  max = 20,
                  value = 2),
      radioButtons(inputId = "householdIllness",
                   label = "Did anyone else in your household fall ill?",
                   choices = c("No"= 0, "Yes" = 1), inline = TRUE),
      conditionalPanel(
        condition = "input.householdIllness == 1",
        sliderInput(inputId = "numberIllInHousehold",
                  label = "How many people in your household fell ill?",
                  min = 0,
                  max = 20,
                  value = 0)),
      radioButtons(inputId = "high_risk_exposure_occupation",
                   label = "Do you work in a high-risk exposure occupation?",
                   choices =  c( "No"= 0,"Yes" = 1), inline = TRUE),
     radioButtons(inputId = "high_risk_interactions",
                 label = "Have you had contact with a confirmed COVID case in the days leading to your test?",
                 choices =  c( "No"= 0,"Yes" = 1), inline = TRUE),
      selectInput(inputId = "country",
                  label = "Which country do you live in?",
                  choices = unique(countries$country_name),
                  selected = "United Kingdom"),
      selectInput(inputId = "region",
                  label = "Which region do you live in?",
                  choices = countries_list,
                  selected="Main territory"),
      uiOutput("region")
      #submitButton("Calculate", icon("refresh"))
    ),
    
    
    
    # Main panel for displaying outputs ----
    mainPanel(
      
      # Output: Histogram ----
      tabsetPanel(
        tabPanel("Disclaimer", htmlOutput("disclaimer")),
        tabPanel("Plot", htmlOutput("distRes"), plotOutput("distPlot")),
        #tabPanel("Table", tableOutput("probs")),
        tabPanel("Report", h3(htmlOutput("Report"))))
      
    )
  )
)


reticulate::source_python('python_script4.py')
# Define server logic required to draw a histogram ----
server <- function(input, output, session) {
  
  # Histogram of the Old Faithful Geyser Data ----
  # with requested number of bins
  # This expression that generates a histogram is wrapped in a call
  # to renderPlot to indicate that:
  #
  # 1. It is "reactive" and therefore should be automatically
  #    re-executed when inputs (input$bins) change
  # 2. Its output type is a plot
  # ------------------ App virtualenv setup (Do not edit) ------------------- #
  
  observe({
    x <- input$country
    
    # Can use character(0) to remove all choices
    if (is.null(x))
      x <- character(0)
    
    # Can also set the label and select items
    country_regions = dplyr::filter(countries, country_name == input$country)
    updateSelectInput(session, "region",
                      label = "What region do you live in?",
                      choices =  country_regions$region,
                      selected = "Main territory"
    )
  })

  
  #output$probs <- renderDataTable({
  #output$region = renderUI({
  #  country_regions = dplyr::filter(regions, country == input$country)
  #  selectInput('region2', 'What region do you live in?', country_regions$region)
  #})
  
  
  dataInput <- reactive({
  if (is.null(input[["date"]])){
    prevalences = fetch_all_prevalence(input$country, input$region, as.Date(Sys.Date()))
  }else{
    prevalences = fetch_all_prevalence(input$country, input$region, input$date)
  }
  print(prevalences)
    
    
    testMethod(achyJointsMuscles= as.numeric(input$myalgia),
               runny_nose = as.numeric(input$runny_nose),
               cough = as.numeric(input$Cough),
               fatigue = as.numeric(input$fatigue),
               fever = as.numeric(input$fever),
               headache = as.numeric(input$headache), 
               lossTaste = as.numeric(input$lossTaste),
               lossSmell = as.numeric(input$lossSmell),
               shortnessOfBreath = as.numeric(input$shortnessOfBreath), 
               soreThroat = as.numeric(input$soreThroat), 
               stomachUpsetDiarrhoea = as.numeric(input$stomachUpsetDiarrhoea),
               householdIllness = as.numeric(input$householdIllness),
               numberInHousehold = as.numeric(input$numberInHousehold), 
               numberIllInHousehold = as.numeric(input$numberIllInHousehold), 
               percentage_householdIllness = as.numeric(input$numberIllInHousehold)/(1+as.numeric(input$numberInHousehold)),
               fever_severity = as.numeric(input$fever_severity),
               cough_severity = as.numeric(input$cough_severity),
               howShortOfBreath = as.numeric(input$howShortOfBreath),
               high_risk_exposure_occupation = as.numeric(input$high_risk_exposure_occupation),
               high_risk_interactions = as.numeric(input$high_risk_interactions),
               date_test = input$date_test,
               type_test  = input$type_test,
               test = as.numeric(input$test),
               symptom_onset = as.numeric(input$symptom_onset),
               symptomatic = as.numeric(input$symptomatic),
               country= input$country,
               region = input$region,
               prev = prevalences$prev,
               pct_ili = prevalences$pct_ili,
               pct_cli = prevalences$pct_cli,
               pct_tested = prevalences$pct_tested_fb,
               sd_prev =  prevalences$sd_prev,
               sd_ili = prevalences$sd_pct_ili ,
               sd_cli = prevalences$sd_pct_cli,
               sd_tested = prevalences$sd_pct_tested_fb
    )
  })
  

  
  output$disclaimer = renderUI({
    tags$div(
      tags$p("This questionnaire uses your personal information and your symptoms history to evaluate your probability of having COVID. Your answers can be combined with your COVID test result to yield a personalized and better-informed probability of being infected."), 
      tags$p('Indeed, COVID test results are imperfect (even the gold standard RT-PCR test -- see below how the sensitivity (that is, the probability of the test accurately detecting the virus if you are infected), varies with time.'),
      tags$img(src = "sensitivity.png", height = 500, width = 702),
      tags$p(""),
      'This image was taken from a recent paper by', tags$a(href ="https://github.com/HopkinsIDD/covidRTPCR", "Kurcinka et al."),
      tags$p('Our method has been designed in collaboration with medical experts. It leverages large public datasets to refine your probability of having COVID, given risk-factors, geographical, temporal data. However, we are not medical experts ourselves. Do not rely on this tool for medical advice.  This tool is purely informative, in controlling the uncertainty in the test performance and symptoms.'),
      tags$p(""),
      'This platform is currently only for COVID test (PCR/rapid test) to diagnose current infection. If you have taken an antibody test (assessing immunity), while we work on merging the two, we refer you to our ', tags$a(href ="https://homecovidtests.shinyapps.io/COVID-app/", "other platform."),
      tags$p("We do not save any of your personal information, nor answers to this questionnaire."),
      tags$p("To use this calculator, please fill out all of the questions on the left hand side, then click on the 'plot' or 'report' tabs at the top of the screen to see your results.")
    )
   })
  
  output$distPlot <- renderPlot({
    x =  dataInput()

    hist(100*x, breaks = seq(from=0, to=100, by=2.5), col = "#75AADB", border = "white",
         xlab = "Probability (in %)",
         main ="Your probability distribution" )
  })
  
  output$distRes <- renderText({
    x =  dataInput()
    #### Write the report
    q = quantile(x, probs =c(0.1, 0.5, 0.90))
    q[2] = mean(x)
    q = 100 *q
    conclusion = "<div style='background-color:red'>"
    if (q[2]> 50){
      conclusion = paste0("According to your questionnaire,  you are more likely to have COVID-19 (probability= ", round(q[2],2),  "%). \n \n")
      if(q[1] < 50){
        conclusion = paste0( conclusion, "The uncertainty associated with this result is high (CI: [" , round(q[1],2), "," , round(q[3],2), "%])." )
      }else{
        conclusion = paste0(conclusion, "The evidence provided by your answers is significant, and the algorithm is fairly confident (CI: [" , round(q[1],2), "," , round(q[3],2), "%])." )
      }
    }else{
      conclusion = paste0("According to your questionnaire, you are more likely to NOT have COVID-19 specific antibodies (probability = ", round(q[2],2),  "%). \n \n")
      if (q[3] > 50){
        conclusion = paste0(conclusion, "The uncertainty associated with this result is high (CI: [" , round(q[1],2), "," , round(q[3],2), "%])." )
      }else{
        conclusion = paste0(conclusion, "The evidence provided by your answers is significant, and the algorithm is fairly confident (CI: [" , round(q[1],2), "," , round(q[3],2), "%])." )
      }
    }
    conclusion = paste0(conclusion, "</div>")
    HTML(conclusion)
  })
  
  symptoms_response<-reactive({
    #sprint(input$symptomatic)
    inc = 0
    inc = as.numeric(input$myalgia) + as.numeric(input$runny_nose) +
          as.numeric(input$Cough)+ as.numeric(input$fatigue) + as.numeric(input$fever) + as.numeric(input$headache) +
          as.numeric(input$lossTaste) +  as.numeric(input$lossSmell)+ as.numeric(input$shortnessOfBreath) + as.numeric(input$soreThroat) + as.numeric(input$stomachUpsetDiarrhoea)
    if((inc == 0) & (input$symptomatic == 0)){
      "You  did not report having felt ill or any specific symptoms."
    }else{
      if((inc == 0) & (input$symptomatic == 1)){
        "You reported having felt ill but did not report any specific symptoms."
      }else{
          "You reported the following symptoms: \n"
      }
    }
  })
  
  test_response<-reactive({
     dplyr::filter(COVID_report, Answer_value == as.numeric(input$test) & Question_ID=="test")
  })
  label_response<-reactive({
    dplyr::filter(COVID_report, Answer_value %in% input$label & Question_ID=="label")
  })
  symptomatic_response<-reactive({
    dplyr::filter(COVID_report, Answer_value == input$symptomatic & Question_ID=="symptomatic")
  })
  chills_response<-reactive({
   t = dplyr::filter(COVID_report, Answer_value %in% input$chills & Question_ID=="chills")
   ifelse(t$Response != "", paste0("-", t$Response, "\n"), "")
  })
  shortnessOfBreath_response<-reactive({
    t= dplyr::filter(COVID_report, Answer_value %in% input$shortnessOfBreath & Question_ID=="shortnessOfBreath")
    ifelse(t$Response != "", paste0("-", t$Response, "\n"), "")
  })
  achyJointsMuscles_response<-reactive({
    t=dplyr::filter(COVID_report, Answer_value %in% input$achyJointsMuscles & Question_ID=="myalgia")
    ifelse(t$Response != "", paste0("-", t$Response, "\n"), "")
  })
  fever_response<-reactive({
    t=dplyr::filter(COVID_report, Answer_value %in% input$fever & Question_ID=="fever")
    ifelse(t$Response != "", paste0("-", t$Response, "\n"), "")
  })
  
  headache_response<-reactive({
    t = dplyr::filter(COVID_report, Answer_value %in% input$headache & Question_ID=="headache")
    ifelse(t$Response != "", paste0("-", t$Response, "\n"), "")
  })
  runnyNose_response<-reactive({
    t = dplyr::filter(COVID_report, Answer_value %in% input$congestedNose & Question_ID=="runny_Nose")
    ifelse(t$Response != "", paste0("-", t$Response, "\n"), "")
  })
  soreThroat_response<-reactive({
    t = dplyr::filter(COVID_report, Answer_value %in% input$soreThroat & Question_ID=="soreThroat")
    ifelse(t$Response != "", paste0("-", t$Response, "\n"), "")
  })
  lossTasteSmell_response<-reactive({
    inc = min(1,as.numeric(input$lossTaste) + as.numeric(input$lossSmell))
    t = dplyr::filter(COVID_report, Answer_value == inc & Question_ID=="lossTasteSmell")
    ifelse(t$Response != "", paste0("-", t$Response, "\n"), "")
  })
  Cough_response<-reactive({
    t = dplyr::filter(COVID_report, Answer_value %in% input$Cough & Question_ID=="Cough")
    ifelse(t$Response != "", paste0("-", t$Response, "\n"), "")
  })
  fatigue_response<-reactive({
    t = dplyr::filter(COVID_report, Answer_value %in% input$fatigue & Question_ID=="fatigue")
    ifelse(t$Response != "", paste0("-", t$Response, "\n"), "")
  })
  stomachUpsetDiarrhoea_response<-reactive({
    t = dplyr::filter(COVID_report, Answer_value %in% input$stomachUpsetDiarrhoea & Question_ID=="stomachUpsetDiarrhoea")
    ifelse(t$Response != "", paste0("-", t$Response, "\n"), "")
  })
  householdIllness_response<-reactive({
    dplyr::filter(COVID_report, Answer_value %in% input$householdIllness & Question_ID=="householdIllness")
  })
  
  output$Report <- renderText({
    gsub(pattern = "\\n", replacement = "<br/>" ,paste0('<p style="font-family:verdana;font-size:14px">',
           toString(test_response()$Response),       
           #toString(label_response()$Response),"\n",
           #toString(symptomatic_response()$Response),"\n",
           # # label_t_response()$Response,"\n",
           toString(symptoms_response()),
           toString(shortnessOfBreath_response()),
            toString(achyJointsMuscles_response()),
            toString(fever_response()),
            toString(headache_response()),
            toString(soreThroat_response()),
            toString(lossTasteSmell_response()),
           # toString(lossSmell_response()),
             toString(Cough_response()),
             toString(fatigue_response()),
             toString(stomachUpsetDiarrhoea_response()),
             toString(runnyNose_response()),
             "\n","\n",
            toString(householdIllness_response()$Response),
     "</p>"))
  })
  
  
}


shinyApp(ui, server)
